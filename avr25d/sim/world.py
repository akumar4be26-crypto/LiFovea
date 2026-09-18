"""Procedural urban world used by the built-in LiDAR simulator.

The world is deliberately *analytic*: the ground is a closed-form height
field and every object is a primitive (axis-aligned box, vertical cylinder,
sphere).  That buys three things the project needs:

* exact per-point ground truth labels, so accuracy numbers are real
  measurements rather than eyeballed guesses;
* a ray caster that is a handful of vectorised NumPy expressions rather
  than a mesh/BVH dependency;
* reproducibility - a seed fully determines a scene.

Coordinate convention: right-handed, +x along the road, +y to the left,
+z up, z = 0 at the nominal road plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ..config import (
    CLASS_DRIVABLE,
    CLASS_DYNAMIC_OBJECT,
    CLASS_ROUGH_TERRAIN,
    CLASS_STATIC_OBSTACLE,
)

# Primitive kind codes
KIND_BOX = 0
KIND_CYLINDER = 1
KIND_SPHERE = 2


@dataclass
class Primitive:
    kind: int
    #: box  -> (xmin, ymin, zmin, xmax, ymax, zmax)
    #: cyl  -> (cx, cy, zmin, radius, zmax, 0)
    #: sph  -> (cx, cy, cz, radius, 0, 0)
    params: np.ndarray
    label: int
    instance: int
    overhang: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)
    reflectivity: float = 0.35

    def moved(self, dt: float) -> "Primitive":
        if self.velocity == (0.0, 0.0):
            return self
        p = self.params.copy()
        vx, vy = self.velocity
        if self.kind == KIND_BOX:
            p[[0, 3]] += vx * dt
            p[[1, 4]] += vy * dt
        else:  # cylinder / sphere share (cx, cy) in slots 0,1
            p[0] += vx * dt
            p[1] += vy * dt
        return Primitive(self.kind, p, self.label, self.instance,
                         self.overhang, self.velocity, self.reflectivity)


@dataclass
class Pothole:
    x: float
    y: float
    radius: float
    depth: float


@dataclass
class World:
    """A stretch of urban road with curbs, potholes, street furniture and
    traffic."""

    length: float = 260.0
    road_half_width: float = 4.0
    curb_height: float = 0.16
    curb_ramp: float = 0.16
    sidewalk_width: float = 2.2
    seed: int = 7

    potholes: List[Pothole] = field(default_factory=list)
    statics: List[Primitive] = field(default_factory=list)
    dynamics: List[Primitive] = field(default_factory=list)
    _noise: np.ndarray = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def generate(cls, seed: int = 7, length: float = 260.0,
                 traffic: int = 9, pedestrians: int = 14) -> "World":
        rng = np.random.default_rng(seed)
        w = cls(length=length, seed=seed)

        # deterministic band-limited noise basis (amplitude, fx, fy, phase)
        w._noise = np.stack([
            rng.uniform(0.006, 0.030, 6),
            rng.uniform(0.03, 0.55, 6),
            rng.uniform(0.03, 0.55, 6),
            rng.uniform(0.0, 2 * np.pi, 6),
        ])

        inst = 1

        # ---- potholes -------------------------------------------------
        for _ in range(16):
            px = rng.uniform(8.0, length - 8.0)
            py = w.road_center(np.array([px]))[0] + rng.uniform(-3.2, 3.2)
            w.potholes.append(Pothole(px, py,
                                      rng.uniform(0.28, 0.75),
                                      rng.uniform(0.07, 0.20)))

        # ---- building facades ----------------------------------------
        for side in (-1.0, 1.0):
            x = 6.0
            while x < length:
                depth = rng.uniform(7.0, 16.0)
                off = w.road_half_width + w.sidewalk_width + rng.uniform(0.8, 2.0)
                y0 = side * off
                y1 = y0 + side * rng.uniform(6.0, 12.0)
                h = rng.uniform(5.0, 16.0)
                w.statics.append(Primitive(
                    KIND_BOX,
                    np.array([x, min(y0, y1), 0.0, x + depth, max(y0, y1), h]),
                    CLASS_STATIC_OBSTACLE, inst, reflectivity=0.45))
                inst += 1
                x += depth + rng.uniform(1.5, 6.0)

        # ---- lamp posts / sign poles ---------------------------------
        for side in (-1.0, 1.0):
            for x in np.arange(11.0, length, 14.0):
                yc = w.road_center(np.array([x]))[0]
                y = yc + side * (w.road_half_width + 1.1)
                w.statics.append(Primitive(
                    KIND_CYLINDER,
                    np.array([x, y, 0.0, 0.085, 7.0, 0.0]),
                    CLASS_STATIC_OBSTACLE, inst, reflectivity=0.55))
                inst += 1

        # ---- street trees: trunk + canopy that overhangs the curb ----
        for side in (-1.0, 1.0):
            for x in np.arange(19.0, length, 23.0):
                yc = w.road_center(np.array([x]))[0]
                y = yc + side * (w.road_half_width + 1.9)
                w.statics.append(Primitive(
                    KIND_CYLINDER,
                    np.array([x, y, 0.0, 0.17, 3.3, 0.0]),
                    CLASS_STATIC_OBSTACLE, inst, reflectivity=0.30))
                inst += 1
                # the canopy hangs over the curb but clears a vehicle: it is
                # an obstacle in plan view and free space in 2.5D
                w.statics.append(Primitive(
                    KIND_SPHERE,
                    np.array([x, y - side * 1.25, 5.35, 2.15, 0.0, 0.0]),
                    CLASS_STATIC_OBSTACLE, inst, overhang=True,
                    reflectivity=0.22))
                inst += 1

        # ---- an overhead gantry: the canonical overhanging obstacle --
        for gx in (58.0, 171.0):
            yc = w.road_center(np.array([gx]))[0]
            w.statics.append(Primitive(
                KIND_BOX,
                np.array([gx, yc - 6.0, 4.6, gx + 0.9, yc + 6.0, 5.3]),
                CLASS_STATIC_OBSTACLE, inst, overhang=True, reflectivity=0.5))
            inst += 1

        # ---- parked + moving vehicles --------------------------------
        for i in range(traffic):
            x = rng.uniform(14.0, length - 20.0)
            yc = w.road_center(np.array([x]))[0]
            lane = rng.choice([-1.0, 1.0])
            y = yc + lane * rng.uniform(1.2, 3.0)
            moving = i % 3 != 0
            vx = float(lane * rng.uniform(3.0, 9.0)) if moving else 0.0
            w.dynamics.append(Primitive(
                KIND_BOX,
                np.array([x - 2.2, y - 0.95, 0.28, x + 2.2, y + 0.95, 1.62]),
                CLASS_DYNAMIC_OBJECT, inst, velocity=(vx, 0.0),
                reflectivity=0.6))
            inst += 1

        # ---- pedestrians ---------------------------------------------
        for i in range(pedestrians):
            x = rng.uniform(10.0, length - 12.0)
            yc = w.road_center(np.array([x]))[0]
            side = rng.choice([-1.0, 1.0])
            crossing = i % 4 == 0
            y = yc + side * (rng.uniform(0.5, 2.0) if crossing
                             else w.road_half_width + rng.uniform(0.4, 2.0))
            vy = float(-side * rng.uniform(0.9, 1.5)) if crossing else 0.0
            vx = float(rng.uniform(-1.2, 1.2)) if not crossing else 0.0
            zbase = 0.0 if crossing else w.curb_height
            w.dynamics.append(Primitive(
                KIND_CYLINDER,
                np.array([x, y, zbase, 0.29, zbase + 1.78, 0.0]),
                CLASS_DYNAMIC_OBJECT, inst, velocity=(vx, vy),
                reflectivity=0.4))
            inst += 1

        return w

    # ------------------------------------------------------------------
    # ground model
    # ------------------------------------------------------------------
    def road_center(self, x: np.ndarray) -> np.ndarray:
        """Lateral position of the road centre-line - the road bends, so a
        curb detector cannot cheat with ``abs(y) > const``."""
        return 1.6 * np.sin(x / 41.0) + 0.7 * np.sin(x / 13.0)

    def _terrain_noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a, fx, fy, ph = self._noise
        out = np.zeros_like(x)
        for i in range(a.shape[0]):
            out += a[i] * np.sin(fx[i] * x + fy[i] * y + ph[i])
        return out

    def ground_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Closed-form terrain elevation (metres)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        u = np.abs(y - self.road_center(x))
        half = self.road_half_width

        # long-wavelength terrain undulation + mild longitudinal grade
        z = 0.30 * np.sin(x / 47.0) + 0.0035 * x + self._terrain_noise(x, y)

        # road crown (drains to the gutter)
        crown = -0.035 * np.clip(u / half, 0.0, 1.0) ** 2
        z = z + np.where(u < half, crown, 0.0)

        # curb: a smoothstep over ``curb_ramp`` metres
        t = np.clip((u - half) / self.curb_ramp, 0.0, 1.0)
        z = z + self.curb_height * (t * t * (3.0 - 2.0 * t))

        # verge beyond the sidewalk rises and gets rough
        v = np.clip((u - (half + self.sidewalk_width)) / 1.5, 0.0, 1.0)
        z = z + v * (0.22 + 0.10 * np.sin(2.9 * x + 3.1 * y))

        # potholes
        for p in self.potholes:
            d2 = (x - p.x) ** 2 + (y - p.y) ** 2
            z = z - p.depth * np.exp(-d2 / (0.5 * p.radius ** 2))
        return z

    def terrain_label(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ground-truth terrain class for a point standing on the ground."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        u = np.abs(y - self.road_center(x))
        drivable = u < (self.road_half_width - 0.05)

        # deep enough potholes are not drivable
        for p in self.potholes:
            d2 = (x - p.x) ** 2 + (y - p.y) ** 2
            deep = p.depth * np.exp(-d2 / (0.5 * p.radius ** 2)) > 0.055
            drivable &= ~deep

        return np.where(drivable, CLASS_DRIVABLE,
                        CLASS_ROUGH_TERRAIN).astype(np.int8)

    # ------------------------------------------------------------------
    def objects_at(self, t: float) -> List[Primitive]:
        """Scene primitives advanced to simulation time ``t`` seconds."""
        return self.statics + [d.moved(t) for d in self.dynamics]

    def ego_pose(self, t: float, speed: float = 6.0,
                 start_x: float = 12.0) -> Tuple[np.ndarray, float]:
        """Sensor position (world) and heading for time ``t``."""
        x = start_x + speed * t
        y = float(self.road_center(np.array([x]))[0]) - 1.7   # right-hand lane
        z = float(self.ground_height(np.array([x]), np.array([y]))[0])
        dx = 1e-2
        yaw = float(np.arctan2(
            self.road_center(np.array([x + dx]))[0] - self.road_center(np.array([x]))[0],
            dx))
        return np.array([x, y, z]), yaw
