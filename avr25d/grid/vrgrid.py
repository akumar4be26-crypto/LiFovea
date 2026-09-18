"""Adaptive variable-resolution 2.5D grid.

The map is a **restricted quadtree anchored to the world**, not a stack of
independent rasters.  Level ``L`` has cell size ``base_cell * 2**L``, so a
level-``L`` cell is exactly four level-``(L-1)`` cells and every boundary at
a coarse level is also a boundary at every finer level.  Two consequences
carry the whole design:

1. **No alignment error.**  A point can only ever fall inside one leaf,
   because the leaves tile the plane exactly.  There is no resampling step
   between resolutions and therefore no place for a half-cell offset to
   creep in.

2. **No data loss at a ring boundary.**  The level of a cell is decided by
   the cell's own geometry - the distance from the sensor to its *nearest
   corner* - never by the distance of an individual point.  Every point in
   a cell therefore resolves to the same level, and a cell that straddles a
   ring boundary refines to the finer level rather than being split between
   two representations.

As the vehicle moves, a cell's correct level changes.  ``rebalance`` migrates
state between levels: coarsening fuses four children by inverse-variance
weighting (information preserving), refinement seeds four children from the
parent with inflated variance (honest about what is actually known).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import NUM_CLASSES, FusionConfig, LodConfig, PipelineConfig

_OFFSET = 1 << 20
_MASK = (1 << 21) - 1

#: bytes actually stored per occupied cell (see :meth:`Grid.memory_bytes`)
BYTES_PER_CELL = (
    8    # int64 key
    + 4  # elevation
    + 4  # variance
    + 4  # z_min
    + 4  # z_max
    + 4  # intensity
    + 4  # n_obs
    + 4 * NUM_CLASSES  # class evidence
    + 4  # dynamic evidence
    + 4  # overhead clearance
    + 4  # last seen frame
)


def pack_keys(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return (((ix + _OFFSET) & _MASK) << 21) | ((iy + _OFFSET) & _MASK)


def unpack_keys(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ix = ((keys >> 21) & _MASK) - _OFFSET
    iy = (keys & _MASK) - _OFFSET
    return ix, iy


# ----------------------------------------------------------------------
@dataclass
class LevelStore:
    """Sparse storage for one quadtree level, kept sorted by key."""

    level: int
    cell: float
    keys: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    z: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    var: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    z_min: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    z_max: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    inten: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    n_obs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    evidence: np.ndarray = field(
        default_factory=lambda: np.zeros((0, NUM_CLASSES), np.float32))
    overhead: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    last_seen: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))

    ARRAYS = ("z", "var", "z_min", "z_max", "inten", "n_obs", "evidence",
              "overhead", "last_seen")

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    # -- lookup --------------------------------------------------------
    def rows(self, keys: np.ndarray) -> np.ndarray:
        """Row index for each key, ``-1`` where absent."""
        if len(self) == 0:
            return np.full(keys.shape[0], -1, dtype=np.int64)
        pos = np.searchsorted(self.keys, keys)
        pos_c = np.clip(pos, 0, len(self) - 1)
        hit = self.keys[pos_c] == keys
        return np.where(hit, pos_c, -1)

    # -- mutation ------------------------------------------------------
    def take(self, idx: np.ndarray) -> "LevelStore":
        out = LevelStore(self.level, self.cell, self.keys[idx])
        for name in self.ARRAYS:
            setattr(out, name, getattr(self, name)[idx])
        return out

    def drop(self, mask_keep: np.ndarray) -> None:
        self.keys = self.keys[mask_keep]
        for name in self.ARRAYS:
            setattr(self, name, getattr(self, name)[mask_keep])

    def append(self, other: "LevelStore") -> None:
        if len(other) == 0:
            return
        self.keys = np.concatenate([self.keys, other.keys])
        for name in self.ARRAYS:
            setattr(self, name,
                    np.concatenate([getattr(self, name), getattr(other, name)]))
        order = np.argsort(self.keys, kind="stable")
        self.keys = self.keys[order]
        for name in self.ARRAYS:
            setattr(self, name, getattr(self, name)[order])

    def centers(self) -> Tuple[np.ndarray, np.ndarray]:
        ix, iy = unpack_keys(self.keys)
        return (ix + 0.5) * self.cell, (iy + 0.5) * self.cell


# ----------------------------------------------------------------------
@dataclass
class CellView:
    """Flat, level-tagged view of the whole map - what gets rendered,
    exported and planned over."""

    level: np.ndarray
    size: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    z: np.ndarray
    var: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    n_obs: np.ndarray
    cls: np.ndarray
    confidence: np.ndarray
    overhead: np.ndarray
    key: np.ndarray

    def __len__(self) -> int:
        return int(self.level.shape[0])


# ----------------------------------------------------------------------
class VariableResolutionGrid:
    """Foveated 2.5D elevation + semantics map."""

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        cfg = cfg or PipelineConfig()
        cfg.validate()
        self.cfg = cfg
        self.lod: LodConfig = cfg.lod
        self.fusion: FusionConfig = cfg.fusion
        self.levels: List[LevelStore] = [
            LevelStore(l, self.lod.cell_size(l)) for l in range(self.lod.n_levels)
        ]
        self.frame_index = 0
        self.stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # level assignment
    # ------------------------------------------------------------------
    def cell_min_distance(self, x: np.ndarray, y: np.ndarray, level: int,
                          sensor_xy: Tuple[float, float]) -> np.ndarray:
        """Distance from the sensor to the *nearest corner* of the level-``L``
        cell containing ``(x, y)``."""
        r = self.lod.cell_size(level)
        x0 = np.floor(x / r) * r
        y0 = np.floor(y / r) * r
        sx, sy = sensor_xy
        dx = np.maximum(np.maximum(x0 - sx, sx - (x0 + r)), 0.0)
        dy = np.maximum(np.maximum(y0 - sy, sy - (y0 + r)), 0.0)
        return np.hypot(dx, dy)

    def leaf_level(self, x: np.ndarray, y: np.ndarray,
                   sensor_xy: Tuple[float, float]) -> np.ndarray:
        """Quadtree leaf level for each position.

        Top-down refinement: start at the coarsest level and descend until
        a cell is close enough to the sensor to deserve subdivision.  The
        test reads the cell, not the point, which is what keeps a ring
        boundary from cutting a cell in half.
        """
        n = x.shape[0]
        level = np.zeros(n, dtype=np.int8)
        decided = np.zeros(n, dtype=bool)
        for L in range(self.lod.n_levels - 1, 0, -1):
            d = self.cell_min_distance(x, y, L, sensor_xy)
            take = (~decided) & (d >= self.lod.ring_radii[L])
            level[take] = L
            decided |= take
            if decided.all():
                break
        return level

    # ------------------------------------------------------------------
    # projection + fusion
    # ------------------------------------------------------------------
    def update(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
               probs: np.ndarray, intensity: np.ndarray,
               sensor_xy: Tuple[float, float], sensor_z: float,
               ground_mask: np.ndarray,
               frame_index: Optional[int] = None) -> Dict[str, float]:
        """Project one classified scan into the map.

        ``ground_mask`` selects the returns that describe the *surface*;
        everything else contributes obstacle evidence and an overhead
        clearance value instead of pulling the elevation estimate up.
        """
        if frame_index is not None:
            self.frame_index = frame_index
        else:
            self.frame_index += 1

        f = self.fusion
        lev = self.leaf_level(x, y, sensor_xy)
        d = np.hypot(x - sensor_xy[0], y - sensor_xy[1])
        touched = 0

        for L, store in enumerate(self.levels):
            sel = lev == L
            if not np.any(sel):
                continue
            r = store.cell
            ix = np.floor(x[sel] / r).astype(np.int64)
            iy = np.floor(y[sel] / r).astype(np.int64)
            keys = pack_keys(ix, iy)

            uniq, inv = np.unique(keys, return_inverse=True)
            m = uniq.shape[0]
            touched += m

            gsel = ground_mask[sel]
            zz = z[sel].astype(np.float64)
            cnt_all = np.bincount(inv, minlength=m)
            cnt_g = np.bincount(inv, weights=gsel.astype(np.float64), minlength=m)

            # --- surface statistics from ground returns only -----------
            zg = np.where(gsel, zz, 0.0)
            sum_g = np.bincount(inv, weights=zg, minlength=m)
            sq_g = np.bincount(inv, weights=zg * zg, minlength=m)
            has_g = cnt_g > 0
            mean_g = np.where(has_g, sum_g / np.maximum(cnt_g, 1), np.nan)
            var_in = np.where(
                cnt_g > 1,
                np.maximum(sq_g / np.maximum(cnt_g, 1) - mean_g ** 2, 0.0),
                0.0)

            # Vertical extent of the *surface*, not of everything in the
            # column.  A road cell under a gantry contains returns 5 m apart;
            # charging that to the surface would declare the road impassable
            # for the one reason a 2.5D map exists to rule out.
            order = np.argsort(inv, kind="stable")
            starts = np.searchsorted(inv[order], np.arange(m))
            zs = zz[order]
            gs = gsel[order]
            zmin_all = np.minimum.reduceat(zs, starts)
            zmax_all = np.maximum.reduceat(zs, starts)
            zmin_g = np.minimum.reduceat(np.where(gs, zs, np.inf), starts)
            zmax_g = np.maximum.reduceat(np.where(gs, zs, -np.inf), starts)
            zmin = np.where(np.isfinite(zmin_g), zmin_g, zmin_all)
            zmax = np.where(np.isfinite(zmax_g), zmax_g, zmax_all)

            # overhead clearance: the lowest non-ground return that is high
            # enough to pass under.  This is the layer a 2D occupancy grid
            # cannot express - a branch at 4 m is not an obstacle.
            nz = (~gsel)[order]
            z_over = np.where(nz, zs, np.inf)
            over = np.minimum.reduceat(z_over, starts)
            over = np.where(np.isfinite(over), over, np.nan)

            inten_mean = np.bincount(
                inv, weights=intensity[sel].astype(np.float64), minlength=m
            ) / np.maximum(cnt_all, 1)

            ev = np.zeros((m, NUM_CLASSES), dtype=np.float64)
            for c in range(NUM_CLASSES):
                ev[:, c] = np.bincount(inv, weights=probs[sel, c].astype(np.float64),
                                       minlength=m)

            # --- measurement variance: sensor noise + quantisation -----
            d_cell = np.bincount(inv, weights=d[sel], minlength=m) / np.maximum(cnt_all, 1)
            sigma = f.z_sigma0 + f.z_sigma_rel * d_cell
            r_meas = sigma ** 2 + f.quantisation_gain * var_in
            r_meas = np.maximum(r_meas / np.maximum(cnt_g, 1), f.min_var)

            self._fuse(store, uniq, mean_g, r_meas, zmin, zmax, over,
                       inten_mean, ev, cnt_all, has_g)

        self.stats = {
            "cells_touched": float(touched),
            "cells_total": float(self.cell_count()),
            "points": float(x.shape[0]),
        }
        return self.stats

    # ------------------------------------------------------------------
    def _fuse(self, store: LevelStore, keys: np.ndarray, meas: np.ndarray,
              r_meas: np.ndarray, zmin: np.ndarray, zmax: np.ndarray,
              over: np.ndarray, inten: np.ndarray, ev: np.ndarray,
              cnt: np.ndarray, has_g: np.ndarray) -> None:
        f = self.fusion
        rows = store.rows(keys)
        known = rows >= 0
        new = ~known

        # ---- existing cells: scalar Kalman update ---------------------
        if np.any(known):
            ri = rows[known]
            prior_var = store.var[ri] + f.process_var
            mk = meas[known]
            rk = r_meas[known]
            valid = has_g[known] & np.isfinite(mk)

            gain = np.where(valid, prior_var / (prior_var + rk), 0.0)
            store.z[ri] = np.where(
                valid, store.z[ri] + gain * (np.nan_to_num(mk) - store.z[ri]),
                store.z[ri]).astype(np.float32)
            store.var[ri] = np.maximum((1.0 - gain) * prior_var,
                                       f.min_var).astype(np.float32)
            store.z_min[ri] = np.minimum(store.z_min[ri], zmin[known]).astype(np.float32)
            store.z_max[ri] = np.maximum(store.z_max[ri], zmax[known]).astype(np.float32)
            store.inten[ri] = (0.7 * store.inten[ri] + 0.3 * inten[known]).astype(np.float32)
            store.n_obs[ri] = np.minimum(store.n_obs[ri] + cnt[known], 2 ** 30)
            store.evidence[ri] = (0.85 * store.evidence[ri] + ev[known]).astype(np.float32)
            ok_over = np.isfinite(over[known])
            cur = store.overhead[ri]
            store.overhead[ri] = np.where(
                ok_over, np.where(np.isnan(cur), over[known],
                                  np.minimum(cur, over[known])), cur).astype(np.float32)
            store.last_seen[ri] = self.frame_index

        # ---- brand new cells -----------------------------------------
        if np.any(new):
            k = keys[new]
            fresh = LevelStore(store.level, store.cell, k)
            init_z = np.where(has_g[new] & np.isfinite(meas[new]),
                              np.nan_to_num(meas[new]), zmin[new])
            fresh.z = init_z.astype(np.float32)
            fresh.var = np.maximum(r_meas[new], f.min_var).astype(np.float32)
            fresh.z_min = zmin[new].astype(np.float32)
            fresh.z_max = zmax[new].astype(np.float32)
            fresh.inten = inten[new].astype(np.float32)
            fresh.n_obs = cnt[new].astype(np.int32)
            fresh.evidence = ev[new].astype(np.float32)
            fresh.overhead = over[new].astype(np.float32)
            fresh.last_seen = np.full(k.shape[0], self.frame_index, dtype=np.int32)
            store.append(fresh)

    # ------------------------------------------------------------------
    # level migration
    # ------------------------------------------------------------------
    def rebalance(self, sensor_xy: Tuple[float, float]) -> Dict[str, int]:
        """Move cells to the level their new distance from the sensor calls
        for.  Coarsening fuses four children by inverse-variance weighting;
        refinement seeds children from their parent with inflated variance.
        """
        merged = split = 0
        # ---- coarsen: fine cells that drifted out of their ring -------
        for L in range(self.lod.n_levels - 1):
            store = self.levels[L]
            if len(store) == 0:
                continue
            cx, cy = store.centers()
            want = self.leaf_level(cx, cy, sensor_xy)
            move = want > L
            if not np.any(move):
                continue
            moving = store.take(np.flatnonzero(move))
            store.drop(~move)
            self._coarsen_into(moving, L + 1)
            merged += len(moving)

        # ---- refine: coarse cells that came into a finer ring ---------
        for L in range(self.lod.n_levels - 1, 0, -1):
            store = self.levels[L]
            if len(store) == 0:
                continue
            cx, cy = store.centers()
            want = self.leaf_level(cx, cy, sensor_xy)
            move = want < L
            if not np.any(move):
                continue
            moving = store.take(np.flatnonzero(move))
            store.drop(~move)
            self._refine_into(moving, L - 1)
            split += len(moving)

        return {"coarsened": merged, "refined": split}

    def _coarsen_into(self, src: LevelStore, dst_level: int) -> None:
        dst = self.levels[dst_level]
        ix, iy = unpack_keys(src.keys)
        pkeys = pack_keys(ix >> 1, iy >> 1)
        uniq, inv = np.unique(pkeys, return_inverse=True)
        m = uniq.shape[0]

        prec = 1.0 / np.maximum(src.var.astype(np.float64), 1e-9)
        sp = np.bincount(inv, weights=prec, minlength=m)
        sz = np.bincount(inv, weights=prec * src.z.astype(np.float64), minlength=m)
        z_new = sz / np.maximum(sp, 1e-12)
        var_new = 1.0 / np.maximum(sp, 1e-12)

        agg = LevelStore(dst_level, self.lod.cell_size(dst_level), uniq)
        agg.z = z_new.astype(np.float32)
        agg.var = var_new.astype(np.float32)
        order = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[order], np.arange(m))
        agg.z_min = np.minimum.reduceat(src.z_min[order], starts).astype(np.float32)
        agg.z_max = np.maximum.reduceat(src.z_max[order], starts).astype(np.float32)
        agg.inten = (np.bincount(inv, weights=src.inten.astype(np.float64),
                                 minlength=m)
                     / np.maximum(np.bincount(inv, minlength=m), 1)).astype(np.float32)
        agg.n_obs = np.bincount(inv, weights=src.n_obs.astype(np.float64),
                                minlength=m).astype(np.int32)
        agg.evidence = np.stack([
            np.bincount(inv, weights=src.evidence[:, c].astype(np.float64),
                        minlength=m) for c in range(NUM_CLASSES)], axis=1).astype(np.float32)
        ov = np.where(np.isnan(src.overhead), np.inf, src.overhead)[order]
        omin = np.minimum.reduceat(ov, starts)
        agg.overhead = np.where(np.isfinite(omin), omin, np.nan).astype(np.float32)
        agg.last_seen = np.maximum.reduceat(src.last_seen[order], starts).astype(np.int32)

        self._merge_store(dst, agg)

    def _refine_into(self, src: LevelStore, dst_level: int) -> None:
        dst = self.levels[dst_level]
        ix, iy = unpack_keys(src.keys)
        n = src.keys.shape[0]
        child = LevelStore(dst_level, self.lod.cell_size(dst_level))
        keys = np.empty(4 * n, dtype=np.int64)
        for k, (ox, oy) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            keys[k * n:(k + 1) * n] = pack_keys((ix << 1) + ox, (iy << 1) + oy)
        rep = lambda a: np.tile(a, (4,) + (1,) * (a.ndim - 1))
        child.keys = keys
        child.z = rep(src.z)
        # A parent carries one elevation for ground its children resolve
        # separately, so splitting must not pretend the children are as
        # well known as the parent was: charge the parent's own vertical
        # spread as extra variance.
        spread = np.maximum(src.z_max - src.z_min, 0.0).astype(np.float32)
        child.var = (rep(src.var) + self.fusion.quantisation_gain
                     * rep(spread) ** 2).astype(np.float32)
        child.z_min = rep(src.z_min)
        child.z_max = rep(src.z_max)
        child.inten = rep(src.inten)
        child.n_obs = (rep(src.n_obs) // 4).astype(np.int32)
        child.evidence = (rep(src.evidence) / 4.0).astype(np.float32)
        child.overhead = rep(src.overhead)
        child.last_seen = rep(src.last_seen)

        order = np.argsort(child.keys, kind="stable")
        child.keys = child.keys[order]
        for name in LevelStore.ARRAYS:
            setattr(child, name, getattr(child, name)[order])
        self._merge_store(dst, child)

    @staticmethod
    def _merge_store(dst: LevelStore, incoming: LevelStore) -> None:
        """Insert ``incoming`` into ``dst``, fusing collisions by precision."""
        if len(dst) == 0:
            dst.keys = incoming.keys
            for name in LevelStore.ARRAYS:
                setattr(dst, name, getattr(incoming, name))
            return
        rows = dst.rows(incoming.keys)
        hit = rows >= 0
        if np.any(hit):
            ri = rows[hit]
            pa = 1.0 / np.maximum(dst.var[ri].astype(np.float64), 1e-9)
            pb = 1.0 / np.maximum(incoming.var[hit].astype(np.float64), 1e-9)
            dst.z[ri] = ((pa * dst.z[ri] + pb * incoming.z[hit]) / (pa + pb)).astype(np.float32)
            dst.var[ri] = (1.0 / (pa + pb)).astype(np.float32)
            dst.z_min[ri] = np.minimum(dst.z_min[ri], incoming.z_min[hit])
            dst.z_max[ri] = np.maximum(dst.z_max[ri], incoming.z_max[hit])
            dst.n_obs[ri] += incoming.n_obs[hit]
            dst.evidence[ri] += incoming.evidence[hit]
            a, b = dst.overhead[ri], incoming.overhead[hit]
            dst.overhead[ri] = np.where(np.isnan(a), b,
                                        np.where(np.isnan(b), a, np.minimum(a, b)))
            dst.last_seen[ri] = np.maximum(dst.last_seen[ri], incoming.last_seen[hit])
        if np.any(~hit):
            dst.append(incoming.take(np.flatnonzero(~hit)))

    # ------------------------------------------------------------------
    def prune(self, sensor_xy: Tuple[float, float]) -> int:
        """Drop cells that fell out of range or went stale."""
        removed = 0
        for store in self.levels:
            if len(store) == 0:
                continue
            cx, cy = store.centers()
            d = np.hypot(cx - sensor_xy[0], cy - sensor_xy[1])
            keep = (d <= self.lod.max_range * 1.05) & (
                self.frame_index - store.last_seen <= self.fusion.stale_frames)
            removed += int((~keep).sum())
            store.drop(keep)
        return removed

    # ------------------------------------------------------------------
    # queries / export
    # ------------------------------------------------------------------
    def cell_count(self) -> int:
        return int(sum(len(s) for s in self.levels))

    def memory_bytes(self) -> int:
        return self.cell_count() * BYTES_PER_CELL

    def lookup(self, x: np.ndarray, y: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Finest existing cell covering each position.

        Returns ``(level, row)`` with ``level = -1`` where nothing is
        mapped.  Walking finest-to-coarsest is what lets neighbours at
        different resolutions talk to each other without any explicit
        cross-level bookkeeping.
        """
        n = x.shape[0]
        lvl = np.full(n, -1, dtype=np.int8)
        row = np.full(n, -1, dtype=np.int64)
        todo = np.ones(n, dtype=bool)
        for L, store in enumerate(self.levels):
            if len(store) == 0 or not np.any(todo):
                continue
            idx = np.flatnonzero(todo)
            r = store.cell
            keys = pack_keys(np.floor(x[idx] / r).astype(np.int64),
                             np.floor(y[idx] / r).astype(np.int64))
            rr = store.rows(keys)
            hit = rr >= 0
            if np.any(hit):
                lvl[idx[hit]] = L
                row[idx[hit]] = rr[hit]
                todo[idx[hit]] = False
        return lvl, row

    def elevation_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lvl, row = self.lookup(x, y)
        out = np.full(x.shape[0], np.nan, dtype=np.float32)
        for L, store in enumerate(self.levels):
            m = lvl == L
            if np.any(m):
                out[m] = store.z[row[m]]
        return out

    def cells(self) -> CellView:
        parts = {k: [] for k in
                 ("level", "size", "cx", "cy", "z", "var", "z_min", "z_max",
                  "n_obs", "cls", "confidence", "overhead", "key")}
        for L, store in enumerate(self.levels):
            if len(store) == 0:
                continue
            cx, cy = store.centers()
            ev = store.evidence
            tot = np.maximum(ev.sum(axis=1), 1e-6)
            cls = np.argmax(ev, axis=1).astype(np.int8)
            conf = ev.max(axis=1) / tot
            parts["level"].append(np.full(len(store), L, dtype=np.int8))
            parts["size"].append(np.full(len(store), store.cell, dtype=np.float32))
            parts["cx"].append(cx.astype(np.float32))
            parts["cy"].append(cy.astype(np.float32))
            parts["z"].append(store.z)
            parts["var"].append(store.var)
            parts["z_min"].append(store.z_min)
            parts["z_max"].append(store.z_max)
            parts["n_obs"].append(store.n_obs)
            parts["cls"].append(cls)
            parts["confidence"].append(conf.astype(np.float32))
            parts["overhead"].append(store.overhead)
            parts["key"].append(store.keys)
        if not parts["level"]:
            empty = np.zeros(0)
            return CellView(*[empty] * 13)
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        return CellView(**cat)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.levels = [LevelStore(l, self.lod.cell_size(l))
                       for l in range(self.lod.n_levels)]
        self.frame_index = 0
