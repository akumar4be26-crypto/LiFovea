"""Vectorised ray-cast LiDAR simulator.

Casts ``n_beams x azimuth_steps`` rays against the analytic world, taking the
nearest hit among the ground height field and the scene primitives.  Because
the intersection that produced each return is known exactly, every point
carries a ground-truth class, instance id and overhang flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import SensorConfig
from .world import KIND_BOX, KIND_CYLINDER, KIND_SPHERE, Primitive, World

_EPS = 1e-9


@dataclass
class PointFrame:
    """One LiDAR revolution, expressed in the sensor frame."""

    xyz: np.ndarray            # (N, 3) float32, sensor frame
    intensity: np.ndarray      # (N,)   float32 in [0, 1]
    ring: np.ndarray           # (N,)   int16 beam index
    label: np.ndarray          # (N,)   int8 ground-truth class
    instance: np.ndarray       # (N,)   int32 (0 = ground)
    overhang: np.ndarray       # (N,)   bool ground-truth overhead flag
    origin: np.ndarray         # (3,)   sensor position in world
    yaw: float                 # sensor heading in world
    timestamp: float
    frame_index: int = 0

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def range(self) -> np.ndarray:
        return np.linalg.norm(self.xyz, axis=1)

    @property
    def planar_range(self) -> np.ndarray:
        return np.hypot(self.xyz[:, 0], self.xyz[:, 1])

    def nbytes(self) -> int:
        return int(self.xyz.nbytes + self.intensity.nbytes + self.ring.nbytes)


class LidarSimulator:
    """Spinning multi-beam LiDAR over a :class:`World`."""

    def __init__(self, world: World, cfg: Optional[SensorConfig] = None,
                 seed: int = 7):
        self.world = world
        self.cfg = cfg or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self._build_rays()

    # ------------------------------------------------------------------
    def _build_rays(self) -> None:
        c = self.cfg
        elev = np.deg2rad(np.linspace(c.fov_up_deg, c.fov_down_deg, c.n_beams))
        azim = np.linspace(0.0, 2 * np.pi, c.azimuth_steps, endpoint=False)
        E, A = np.meshgrid(elev, azim, indexing="ij")
        ce = np.cos(E)
        self._dir_local = np.stack(
            [ce * np.cos(A), ce * np.sin(A), np.sin(E)], axis=-1
        ).reshape(-1, 3).astype(np.float64)
        self._ring = np.repeat(np.arange(c.n_beams, dtype=np.int16),
                               c.azimuth_steps)

    # ------------------------------------------------------------------
    def _ground_hit(self, o: np.ndarray, d: np.ndarray) -> np.ndarray:
        """First intersection of each ray with the height field.

        Solves ``o_z + t d_z = H(o_xy + t d_xy)`` by damped fixed-point
        iteration, which converges in a handful of steps for the shallow
        slopes of a road scene and costs one height-field evaluation per
        iteration instead of a full ray march.
        """
        w = self.world
        t = np.full(d.shape[0], np.inf)
        down = d[:, 2] < -1e-4
        if not np.any(down):
            return t

        dz = d[down, 2]
        dx, dy = d[down, 0], d[down, 1]
        h0 = w.ground_height(np.array([o[0]]), np.array([o[1]]))[0]
        tk = np.clip((h0 - o[2]) / dz, 0.0, self.cfg.max_range)

        for _ in range(12):
            px = o[0] + tk * dx
            py = o[1] + tk * dy
            h = w.ground_height(px, py)
            t_new = (h - o[2]) / dz
            tk = tk + 0.8 * (np.clip(t_new, 0.0, 4.0 * self.cfg.max_range) - tk)

        px = o[0] + tk * dx
        py = o[1] + tk * dy
        residual = np.abs(o[2] + tk * dz - w.ground_height(px, py))
        ok = (residual < 0.04) & (tk > self.cfg.min_range) & (tk < self.cfg.max_range)

        tt = np.where(ok, tk, np.inf)
        t[down] = tt
        return t

    # ------------------------------------------------------------------
    @staticmethod
    def _hit_box(o, d, p):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / np.where(np.abs(d) < _EPS, _EPS, d)
            t1 = (p[:3] - o) * inv
            t2 = (p[3:] - o) * inv
            tmin = np.max(np.minimum(t1, t2), axis=1)
            tmax = np.min(np.maximum(t1, t2), axis=1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0.0)
        return np.where(hit, tmin, np.inf)

    @staticmethod
    def _hit_cylinder(o, d, p):
        cx, cy, zmin, r, zmax = p[0], p[1], p[2], p[3], p[4]
        ex, ey = o[0] - cx, o[1] - cy
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2.0 * (d[:, 0] * ex + d[:, 1] * ey)
        c = ex * ex + ey * ey - r * r
        disc = b * b - 4 * a * c
        out = np.full(d.shape[0], np.inf)
        m = (disc > 0) & (a > _EPS)
        if not np.any(m):
            return out
        sq = np.sqrt(disc[m])
        am, bm = a[m], b[m]
        for root in ((-bm - sq) / (2 * am), (-bm + sq) / (2 * am)):
            z = o[2] + root * d[m, 2]
            good = (root > 0) & (z >= zmin) & (z <= zmax)
            cur = out[m]
            out[m] = np.where(good & (root < cur), root, cur)
        return out

    @staticmethod
    def _hit_sphere(o, d, p):
        cen = p[:3]
        r = p[3]
        e = o - cen
        b = 2.0 * (d @ e)
        c = float(e @ e) - r * r
        disc = b * b - 4.0 * c
        out = np.full(d.shape[0], np.inf)
        m = disc > 0
        if not np.any(m):
            return out
        sq = np.sqrt(disc[m])
        t0 = (-b[m] - sq) / 2.0
        t1 = (-b[m] + sq) / 2.0
        t = np.where(t0 > 0, t0, t1)
        out[m] = np.where(t > 0, t, np.inf)
        return out

    @staticmethod
    def _prim_bounds(p: Primitive) -> np.ndarray:
        q = p.params
        if p.kind == KIND_BOX:
            return np.array([q[0], q[1], q[3], q[4]])
        if p.kind == KIND_CYLINDER:
            return np.array([q[0] - q[3], q[1] - q[3], q[0] + q[3], q[1] + q[3]])
        return np.array([q[0] - q[3], q[1] - q[3], q[0] + q[3], q[1] + q[3]])

    # ------------------------------------------------------------------
    def scan(self, t_sim: float, frame_index: int = 0,
             speed: float = 6.0) -> PointFrame:
        cfg = self.cfg
        o_ground, yaw = self.world.ego_pose(t_sim, speed=speed)
        origin = o_ground + np.array([0.0, 0.0, cfg.mount_height])

        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        d = self._dir_local @ R.T

        t_best = self._ground_hit(origin, d)
        t_fin = np.where(np.isfinite(t_best), t_best, 0.0)
        label = self.world.terrain_label(origin[0] + t_fin * d[:, 0],
                                         origin[1] + t_fin * d[:, 1])
        label = np.where(np.isfinite(t_best), label, 0).astype(np.int8)
        inst = np.zeros(d.shape[0], dtype=np.int32)
        over = np.zeros(d.shape[0], dtype=bool)
        refl = np.full(d.shape[0], 0.28, dtype=np.float64)

        objs: List[Primitive] = self.world.objects_at(t_sim)
        for p in objs:
            b = self._prim_bounds(p)
            # cheap circular cull against the sensor's range envelope
            dxm = max(b[0] - origin[0], origin[0] - b[2], 0.0)
            dym = max(b[1] - origin[1], origin[1] - b[3], 0.0)
            if dxm * dxm + dym * dym > cfg.max_range ** 2:
                continue
            if p.kind == KIND_BOX:
                th = self._hit_box(origin, d, p.params)
            elif p.kind == KIND_CYLINDER:
                th = self._hit_cylinder(origin, d, p.params)
            else:
                th = self._hit_sphere(origin, d, p.params)
            better = th < t_best
            if not np.any(better):
                continue
            t_best = np.where(better, th, t_best)
            label = np.where(better, p.label, label).astype(np.int8)
            inst = np.where(better, p.instance, inst)
            over = np.where(better, p.overhang, over)
            refl = np.where(better, p.reflectivity, refl)

        valid = np.isfinite(t_best) & (t_best > cfg.min_range) & (t_best < cfg.max_range)
        if not np.any(valid):
            raise RuntimeError("simulator produced an empty scan")

        t_hit = t_best[valid]
        dv = d[valid]

        # range noise + stochastic dropout
        sigma = cfg.range_sigma0 + cfg.range_sigma_rel * t_hit
        t_noisy = t_hit + self.rng.normal(0.0, sigma)
        keep = self.rng.random(t_noisy.shape[0]) > cfg.dropout
        t_noisy = t_noisy[keep]
        dv = dv[keep]

        world_pts = origin[None, :] + t_noisy[:, None] * dv
        local = (world_pts - origin[None, :]) @ R          # R^T applied on the right

        inten = np.clip(refl[valid][keep] * (35.0 / (t_noisy + 4.0)), 0.02, 1.0)

        return PointFrame(
            xyz=local.astype(np.float32),
            intensity=inten.astype(np.float32),
            ring=self._ring[valid][keep],
            label=label[valid][keep].astype(np.int8),
            instance=inst[valid][keep],
            overhang=over[valid][keep],
            origin=origin,
            yaw=float(yaw),
            timestamp=t_sim,
            frame_index=frame_index,
        )

    # ------------------------------------------------------------------
    def sequence(self, n_frames: int, speed: float = 6.0,
                 start_frame: int = 0):
        """Yield consecutive frames at the sensor's native rate."""
        dt = self.cfg.frame_period
        for i in range(n_frames):
            k = start_frame + i
            yield self.scan(k * dt, frame_index=k, speed=speed)
