"""Deterministic geometric segmenter - the NumPy fallback backend.

This is the path the pipeline takes when PyTorch is unavailable, when no
checkpoint has been trained yet, or when a run has to be bit-for-bit
reproducible.  It is a real classifier, not a stub:

* a lower-envelope ground surface that survives vehicle roofs and walls;
* a *connectivity* test for the drivable/non-drivable split - a patch of
  ground is drivable only if the vehicle can reach it without crossing a
  step taller than it can climb, which is what actually separates a road
  from the sidewalk behind a 14 cm curb (both are perfectly flat);
* BEV connected components with shape priors to split dynamic traffic out
  of static structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import (
    CLASS_DRIVABLE,
    CLASS_DYNAMIC_OBJECT,
    CLASS_ROUGH_TERRAIN,
    CLASS_STATIC_OBSTACLE,
    NUM_CLASSES,
    VehicleConfig,
)
from .features import FeatureBundle, compute_features

try:                                    # optional acceleration only
    from scipy import ndimage as _ndi
except Exception:                       # pragma: no cover
    _ndi = None


# ----------------------------------------------------------------------
# dense helpers on a small BEV raster
# ----------------------------------------------------------------------
def _shifts(a: np.ndarray):
    yield np.roll(a, 1, 0)
    yield np.roll(a, -1, 0)
    yield np.roll(a, 1, 1)
    yield np.roll(a, -1, 1)


def morphological_fill(raster: np.ndarray, valid: np.ndarray,
                       iterations: int = 40) -> np.ndarray:
    """Fill unobserved cells from the nearest observed one.

    With SciPy this is a single exact distance transform; without it, an
    iterative neighbour average converges to much the same surface.
    """
    if not valid.any():
        return np.zeros_like(raster)
    if _ndi is not None:
        _, (ii, jj) = _ndi.distance_transform_edt(~valid, return_indices=True)
        return raster[ii, jj]
    out = raster.copy()
    known = valid.copy()
    for _ in range(iterations):
        if known.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for s, k in zip(_shifts(out), _shifts(known.astype(out.dtype))):
            acc += np.where(k > 0, s, 0.0)
            cnt += k
        newly = (~known) & (cnt > 0)
        out[newly] = acc[newly] / cnt[newly]
        known |= newly
    out[~known] = 0.0
    return out


def connected_components(mask: np.ndarray, max_iter: int = 600) -> np.ndarray:
    """Label 4-connected components of a boolean raster.

    Uses ``scipy.ndimage`` when available and falls back to min-index
    propagation in pure NumPy, so the package has no hard SciPy dependency.
    Unlabelled cells are ``-1``.
    """
    if _ndi is not None:
        lab, _ = _ndi.label(mask)
        return np.where(mask, lab.astype(np.int64), -1)

    h, w = mask.shape
    lab = np.where(mask, np.arange(h * w).reshape(h, w), -1).astype(np.int64)
    big = np.iinfo(np.int64).max
    for _ in range(max_iter):
        cur = np.where(lab >= 0, lab, big)
        nxt = cur.copy()
        for s in _shifts(cur):
            nxt = np.minimum(nxt, s)
        nxt = np.where(mask, nxt, big)
        new_lab = np.where(nxt == big, -1, nxt)
        if np.array_equal(new_lab, lab):
            break
        lab = new_lab
    return lab


# ----------------------------------------------------------------------
@dataclass
class GroundModel:
    raster: np.ndarray        # (H, W) ground elevation (lower envelope)
    observed: np.ndarray      # (H, W) bool - had at least one return
    relief: np.ndarray        # (H, W) height spread of ground returns in-cell
    cell: float
    origin: Tuple[float, float]

    def indices(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.raster.shape
        i = np.clip(((y - self.origin[1]) / self.cell).astype(np.int64), 0, h - 1)
        j = np.clip(((x - self.origin[0]) / self.cell).astype(np.int64), 0, w - 1)
        return i, j

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        i, j = self.indices(x, y)
        return self.raster[i, j]

    def slope_deg(self) -> np.ndarray:
        gy, gx = np.gradient(self.raster, self.cell)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def step_height(self) -> np.ndarray:
        """Largest elevation discontinuity to a 4-neighbour - the curb cue."""
        out = np.zeros_like(self.raster)
        for s in _shifts(self.raster):
            out = np.maximum(out, np.abs(self.raster - s))
        return out


def estimate_ground(xyz: np.ndarray, extent: float = 100.0,
                    cell: float = 0.5) -> GroundModel:
    """Lower-envelope ground surface on a BEV raster."""
    n_side = int(2 * extent / cell) + 1
    origin = (-extent, -extent)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    inside = (np.abs(x) < extent) & (np.abs(y) < extent)
    i = ((y[inside] - origin[1]) / cell).astype(np.int64)
    j = ((x[inside] - origin[0]) / cell).astype(np.int64)
    flat = i * n_side + j

    zz = z[inside].astype(np.float64)
    order = np.argsort(flat, kind="stable")
    fs, zs = flat[order], zz[order]
    uniq, starts = np.unique(fs, return_index=True)
    mins = np.minimum.reduceat(zs, starts)

    # in-cell relief of the *ground-like* returns: the vertical spread of
    # everything within 40 cm of the cell's lowest return.  A cell that
    # contains a curb lip or a pothole rim shows relief even though its
    # neighbours agree with it, and that is exactly the cell a vehicle
    # cannot drive across.
    cell_low = np.repeat(mins, np.diff(np.append(starts, zs.shape[0])))
    near_ground = zs < cell_low + 0.40
    zs_g = np.where(near_ground, zs, -np.inf)
    gmax = np.maximum.reduceat(zs_g, starts)
    relief_vals = np.where(np.isfinite(gmax), gmax - mins, 0.0)

    raster = np.zeros((n_side, n_side), dtype=np.float64)
    observed = np.zeros((n_side, n_side), dtype=bool)
    relief = np.zeros((n_side, n_side), dtype=np.float64)
    raster.flat[uniq] = mins
    relief.flat[uniq] = relief_vals
    observed.flat[uniq] = True
    raster = morphological_fill(raster, observed)

    # two sweeps of lower-envelope smoothing: kills returns that sit on top
    # of a structure yet are the only return in their column
    for _ in range(2):
        nb = np.stack(list(_shifts(raster)))
        raster = np.minimum(raster, np.median(nb, axis=0) + 0.30)
    return GroundModel(raster=raster, observed=observed, relief=relief,
                       cell=cell, origin=origin)


def edge_aware_components(passable: np.ndarray, height: np.ndarray,
                          max_step: float) -> np.ndarray:
    """Connected components where *edges*, not cells, carry the step test.

    Blocking whole cells whenever they touch a curb also severs the road at
    every pothole rim.  Here the graph keeps both cells, and only the edge
    that climbs the curb is cut.  The trick is to label a grid of twice the
    resolution in which even indices are cells and odd indices are the
    edges between them - then any ordinary connected-component labeller
    respects the cut edges.
    """
    h, w = passable.shape
    big = np.zeros((2 * h - 1, 2 * w - 1), dtype=bool)
    big[0::2, 0::2] = passable

    dv = np.abs(height[1:, :] - height[:-1, :]) <= max_step
    big[1::2, 0::2] = passable[1:, :] & passable[:-1, :] & dv

    dh = np.abs(height[:, 1:] - height[:, :-1]) <= max_step
    big[0::2, 1::2] = passable[:, 1:] & passable[:, :-1] & dh

    lab_big = connected_components(big)
    return np.where(passable, lab_big[0::2, 0::2], -1)


def range_adaptive_support(ground: "GroundModel",
                           thresholds: Tuple[float, ...] = (16.0, 30.0, 50.0, 75.0)
                           ) -> np.ndarray:
    """Cells the sensor has actually sampled, closed over the gaps that beam
    divergence opens up at range.

    A fixed morphological closing would either leave holes near the sensor
    or hallucinate surface far away.  The closing radius here grows with
    distance, matching the way a beam's along-range footprint grows - the
    same principle that motivates the variable resolution grid.
    """
    h, w = ground.raster.shape
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (jj + 0.5) * ground.cell + ground.origin[0]
    cy = (ii + 0.5) * ground.cell + ground.origin[1]
    r = np.hypot(cx, cy)

    sup = ground.observed.copy()
    for thr in thresholds:
        grown = sup.copy()
        for s in _shifts(sup):
            grown |= s
        sup = sup | (grown & (r > thr))
    return sup


def drivable_region(ground: GroundModel, vehicle: VehicleConfig,
                    seed_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Ground cells the vehicle can actually reach.

    A cell is *passable* if its local slope and its step to the neighbours
    are within the platform's limits; the drivable set is then the passable
    component that contains the vehicle.  Flat ground behind a curb fails
    not because it is rough but because it is unreachable - exactly the
    distinction a 2D occupancy grid throws away.
    """
    slope = ground.slope_deg()
    passable = (slope <= vehicle.max_slope_deg) & range_adaptive_support(ground)
    passable &= ground.relief <= vehicle.max_step_height

    lab = edge_aware_components(passable, ground.raster, vehicle.max_step_height)
    i, j = ground.indices(np.array([seed_xy[0]]), np.array([seed_xy[1]]))
    seed = lab[i[0], j[0]]
    if seed < 0:
        # the vehicle stands on a cell that failed the test (e.g. it is
        # straddling a pothole): fall back to the largest passable blob
        ids, counts = np.unique(lab[lab >= 0], return_counts=True)
        if ids.size == 0:
            return passable
        seed = ids[np.argmax(counts)]
    return lab == seed


def _footprint_shape(hx, hy, order, starts, counts):
    """Per-cluster BEV footprint thickness and length via 2x2 PCA."""
    xs, ys = hx[order], hy[order]
    n = counts.astype(np.float64)
    sx = np.add.reduceat(xs, starts)
    sy = np.add.reduceat(ys, starts)
    sxx = np.add.reduceat(xs * xs, starts)
    syy = np.add.reduceat(ys * ys, starts)
    sxy = np.add.reduceat(xs * ys, starts)
    mx, my = sx / n, sy / n
    cxx = sxx / n - mx * mx
    cyy = syy / n - my * my
    cxy = sxy / n - mx * my
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l_hi = np.maximum(tr / 2.0 + disc, 0.0)
    l_lo = np.maximum(tr / 2.0 - disc, 0.0)
    return 2.0 * np.sqrt(l_lo), 4.0 * np.sqrt(l_hi)


# ----------------------------------------------------------------------
class GeometricSegmenter:
    """Rule-based four-class point segmenter."""

    name = "geometric"

    def __init__(self, vehicle: Optional[VehicleConfig] = None,
                 extent: float = 100.0, ground_cell: float = 0.5,
                 fov_up_deg: float = 2.0):
        self.vehicle = vehicle or VehicleConfig()
        self.extent = extent
        self.ground_cell = ground_cell
        self.fov_up_deg = fov_up_deg

    # ------------------------------------------------------------------
    def predict(self, xyz: np.ndarray, intensity: np.ndarray,
                bundle: Optional[FeatureBundle] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        if bundle is None:
            bundle = compute_features(xyz, intensity)
        v = self.vehicle
        n = xyz.shape[0]
        f = bundle.features

        ground = estimate_ground(xyz, extent=self.extent, cell=self.ground_cell)
        gz = ground.sample(xyz[:, 0], xyz[:, 1])
        height = xyz[:, 2] - gz

        normal_z = f[:, 10]
        roughness = f[:, 11]
        scattering = f[:, 8]
        linearity = f[:, 6]

        is_ground = ((height < 0.20) & (normal_z > 0.50)) | (height < 0.07)

        # ---- preliminary terrain split -------------------------------
        # Only *local* evidence is used here - in-cell relief, slope and
        # surface roughness.  The reachability half of the question ("flat,
        # but on the far side of a curb") needs a resolution that varies
        # with range, so it is answered later by the 2.5D grid, which then
        # writes the refined terrain class back onto these points.
        gi, gj = ground.indices(xyz[:, 0], xyz[:, 1])
        slope = ground.slope_deg()
        pt_relief = ground.relief[gi, gj]
        pt_slope = slope[gi, gj]
        observed_here = ground.observed[gi, gj]

        rough_local = (
            (pt_relief > v.max_step_height)
            | (pt_slope > v.max_slope_deg)
            | (roughness > v.max_roughness * 1.8)
        ) & observed_here

        labels = np.where(
            is_ground,
            np.where(rough_local, CLASS_ROUGH_TERRAIN, CLASS_DRIVABLE),
            CLASS_STATIC_OBSTACLE,
        ).astype(np.int8)

        # ---- dynamic objects: BEV clustering + shape priors ----------
        obj = (~is_ground) & (height > 0.25) & (height < 4.0)
        if np.any(obj):
            cell = 0.5
            n_side = int(2 * self.extent / cell) + 1
            oi = np.clip(((xyz[obj, 1] + self.extent) / cell).astype(np.int64),
                         0, n_side - 1)
            oj = np.clip(((xyz[obj, 0] + self.extent) / cell).astype(np.int64),
                         0, n_side - 1)
            occ = np.zeros((n_side, n_side), dtype=bool)
            occ[oi, oj] = True
            # close one-cell gaps first, otherwise a facade that a tree
            # occludes shatters into vehicle-sized fragments
            closed = occ.copy()
            for s in _shifts(occ):
                closed |= s
            lab = connected_components(closed)
            pt_comp = lab[oi, oj]

            comp_ids, comp_inv, comp_cnt = np.unique(pt_comp, return_inverse=True,
                                                     return_counts=True)
            hx, hy, hz = xyz[obj, 0], xyz[obj, 1], height[obj]
            k = comp_ids.shape[0]
            order = np.argsort(comp_inv, kind="stable")
            starts = np.searchsorted(comp_inv[order], np.arange(k))

            def _span(vals):
                vs = vals[order]
                return (np.maximum.reduceat(vs, starts)
                        - np.minimum.reduceat(vs, starts))

            ex, ey = _span(hx), _span(hy)
            top = np.maximum.reduceat(hz[order], starts)
            base = np.minimum.reduceat(hz[order], starts)
            foot = np.maximum(ex, ey)
            small = np.minimum(ex, ey)

            # Close to the vehicle the sensor's vertical field of view
            # clips a building facade, so the facade *looks* van-sized in
            # height.  What still separates them is the footprint: a facade
            # is a plane, a vehicle is a box.  Fit a line to each cluster's
            # BEV footprint and measure the residual thickness.
            thickness, extent_len = _footprint_shape(hx, hy, order, starts,
                                                     comp_cnt)
            wall_like = (thickness < 0.30) & (extent_len > 2.5)

            pedestrian = (foot < 1.5) & (top > 0.9) & (top < 2.3) & (comp_cnt > 5)
            vehicle_like = ((foot > 1.6) & (foot < 7.0) & (small < 3.2)
                            & (top > 0.7) & (top < 2.6) & (comp_cnt > 20))
            grounded = base < 0.75
            dyn_comp = (pedestrian | vehicle_like) & grounded & ~wall_like

            sel = dyn_comp[comp_inv]
            sel &= ~((scattering[obj] > 0.30) & (hz > 2.3))   # foliage
            idx = np.flatnonzero(obj)
            labels[idx[sel]] = CLASS_DYNAMIC_OBJECT

        # thin vertical structure is always static (poles, sign posts)
        labels[(linearity > 0.85) & (height > 1.2)] = CLASS_STATIC_OBSTACLE

        probs = np.full((n, NUM_CLASSES), 0.04, dtype=np.float32)
        probs[np.arange(n), labels] = 0.88
        probs /= probs.sum(axis=1, keepdims=True)
        return labels, probs
