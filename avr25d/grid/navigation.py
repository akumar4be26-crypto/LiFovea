"""Uniform-resolution navigation layer built on top of the 2.5D map.

Perception wants the finest resolution the sensor can support; connectivity
and planning want a graph whose nodes are roughly the size of the vehicle.
Those are different jobs, and forcing one structure to do both is what makes
naive multi-resolution maps fall over: at 5 cm the returns from a single
revolution are sparse islands, so any reachability test on them collapses.

So the map keeps its variable resolution, and this layer aggregates it into
uniform ``nav_cell`` tiles - a few thousand nodes instead of a hundred
thousand - carrying exactly what a planner needs:

* representative elevation, and the *internal relief* of the tile, which is
  what a curb looks like once you stop resolving it;
* whether anything in the tile blocks the vehicle at body height;
* neighbour links that probe across the radial sampling gaps of a spinning
  sensor, so a tile is connected to the next observed tile rather than to
  nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import VehicleConfig
from .vrgrid import CellView, pack_keys, unpack_keys

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
except Exception:                                   # pragma: no cover
    coo_matrix = None
    _cc = None

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class NavLayer:
    cell: float
    keys: np.ndarray            # (M,) sorted packed keys
    cx: np.ndarray
    cy: np.ndarray
    z: np.ndarray               # representative surface elevation
    relief: np.ndarray          # internal ground relief  (curb detector)
    blocked: np.ndarray         # something solid at body height
    has_ground: np.ndarray
    n_cells: np.ndarray
    parent: np.ndarray          # (N_map,) nav index of every map cell
    overhead_only: np.ndarray = None   # carries structure that clears the roof
    passable: np.ndarray = None
    reachable: np.ndarray = None
    neighbours: np.ndarray = None
    neighbour_dist: np.ndarray = None
    slope_deg: np.ndarray = None
    step: np.ndarray = None

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def rows(self, keys: np.ndarray) -> np.ndarray:
        if len(self) == 0:
            return np.full(keys.shape[0], -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(self.keys, keys), 0, len(self) - 1)
        return np.where(self.keys[pos] == keys, pos, -1)


def _tile_pairs(cv: CellView, nav_cell: float) -> Tuple[np.ndarray, np.ndarray]:
    """(tile key, source cell) pairs.

    Map cells finer than a tile fall into exactly one tile.  Map cells
    *coarser* than a tile - the far field, where cells reach 80 cm - cover
    several tiles, and each of those tiles has to inherit the cell, or the
    navigation layer would be full of holes precisely where the map is
    coarsest.  Because both grids are powers of two of the same base cell,
    the cover is exact: no interpolation, no partial tiles.
    """
    factor = np.maximum(np.round(cv.size / nav_cell).astype(np.int64), 1)
    keys_out, src_out = [], []
    for f in np.unique(factor):
        sel = np.flatnonzero(factor == f)
        if f == 1:
            ix = np.floor(cv.cx[sel] / nav_cell).astype(np.int64)
            iy = np.floor(cv.cy[sel] / nav_cell).astype(np.int64)
            keys_out.append(pack_keys(ix, iy))
            src_out.append(sel)
            continue
        ix0 = np.floor((cv.cx[sel] - 0.5 * cv.size[sel] + 1e-6) / nav_cell).astype(np.int64)
        iy0 = np.floor((cv.cy[sel] - 0.5 * cv.size[sel] + 1e-6) / nav_cell).astype(np.int64)
        for a in range(int(f)):
            for b in range(int(f)):
                keys_out.append(pack_keys(ix0 + a, iy0 + b))
                src_out.append(sel)
    return np.concatenate(keys_out), np.concatenate(src_out)


def build(cv: CellView, local_ok: np.ndarray, obstacle: np.ndarray,
          nav_cell: float = 0.4, clearance: float = 2.4) -> NavLayer:
    """Aggregate map cells into uniform navigation tiles.

    An obstacle only blocks a tile if part of it hangs below the vehicle's
    clearance *above that tile's own ground*.  A gantry at 4.6 m and a wall
    both paint the same footprint in plan view; the difference only exists
    in 2.5D, and this is where it is spent.
    """
    n = len(cv)
    keys, src = _tile_pairs(cv, nav_cell)
    uniq, inv = np.unique(keys, return_inverse=True)
    m = uniq.shape[0]
    cvz = cv.z[src]
    cv_base = cv.z_min[src]
    local_ok = local_ok[src]
    obstacle = obstacle[src]

    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(m))

    ground = local_ok & ~obstacle
    big = np.float32(1e9)
    zg_hi = np.where(ground, cvz, -big)[order]
    zg_lo = np.where(ground, cvz, big)[order]
    zmax = np.maximum.reduceat(zg_hi, starts)
    zmin = np.minimum.reduceat(zg_lo, starts)
    cnt_g = np.bincount(inv, weights=ground.astype(np.float64), minlength=m)
    has_ground = cnt_g > 0

    zsum = np.bincount(inv, weights=np.where(ground, cvz, 0.0).astype(np.float64),
                       minlength=m)
    z = np.where(has_ground, zsum / np.maximum(cnt_g, 1), np.nan)
    relief = np.where(has_ground, zmax - zmin, 0.0)

    # second pass: which obstacles actually reach down into the vehicle's
    # envelope, measured against the tile's own ground elevation
    ref = np.where(has_ground, z, -np.inf)[inv]
    low = obstacle & (cv_base < ref + clearance)
    blk = np.bincount(inv, weights=low.astype(np.float64), minlength=m) > 0
    overhead_only = np.bincount(inv, weights=(obstacle & ~low).astype(np.float64),
                                minlength=m) > 0

    kx, ky = unpack_keys(uniq)
    layer = NavLayer(
        cell=nav_cell,
        keys=uniq,
        cx=((kx + 0.5) * nav_cell).astype(np.float32),
        cy=((ky + 0.5) * nav_cell).astype(np.float32),
        z=z.astype(np.float32),
        relief=relief.astype(np.float32),
        blocked=blk,
        has_ground=has_ground,
        n_cells=np.bincount(inv, minlength=m).astype(np.int32),
        parent=np.zeros(0, dtype=np.int64),
        overhead_only=overhead_only,
    )
    # every map cell is owned by the tile containing its centre
    own = pack_keys(np.floor(cv.cx / nav_cell).astype(np.int64),
                    np.floor(cv.cy / nav_cell).astype(np.int64))
    layer.parent = layer.rows(own)
    return layer


def link(nav: NavLayer, vehicle: VehicleConfig, max_probe: int = 10) -> NavLayer:
    """Neighbour links, slope/step, and the local passability verdict."""
    m = len(nav)
    ix, iy = unpack_keys(nav.keys)
    out = np.full((m, 4), -1, dtype=np.int64)
    dist = np.full((m, 4), np.inf, dtype=np.float32)
    for k, (dx, dy) in enumerate(_DIRS):
        todo = np.ones(m, dtype=bool)
        for s in range(1, max_probe + 1):
            idx = np.flatnonzero(todo)
            if idx.size == 0:
                break
            probe = pack_keys(ix[idx] + dx * s, iy[idx] + dy * s)
            r = nav.rows(probe)
            hit = r >= 0
            if np.any(hit):
                tgt = idx[hit]
                out[tgt, k] = r[hit]
                dist[tgt, k] = s * nav.cell
                todo[tgt] = False
    nav.neighbours = out
    nav.neighbour_dist = dist

    step = np.zeros(m, dtype=np.float32)
    slope = np.zeros(m, dtype=np.float32)
    for k in range(4):
        idx = out[:, k]
        ok = (idx >= 0) & nav.has_ground & nav.has_ground[np.maximum(idx, 0)]
        if not np.any(ok):
            continue
        dz = np.abs(nav.z[ok] - nav.z[idx[ok]])
        d = np.maximum(dist[ok, k], 1e-3)
        step[ok] = np.maximum(step[ok], dz)
        slope[ok] = np.maximum(slope[ok], np.degrees(np.arctan2(dz, d)))
    nav.step = step
    nav.slope_deg = slope
    nav.passable = (nav.has_ground & ~nav.blocked
                    & (nav.relief <= vehicle.max_step_height)
                    & (slope <= vehicle.max_slope_deg))
    return nav


def reachability(nav: NavLayer, vehicle: VehicleConfig,
                 seed_xy: Tuple[float, float],
                 seed_radius: float = 14.0,
                 max_link: float = 2.0) -> np.ndarray:
    """Tiles connected to the vehicle across edges it can cross."""
    m = len(nav)
    if m == 0:
        return np.zeros(0, dtype=bool)
    rows, cols = [], []
    for k in range(4):
        idx = nav.neighbours[:, k]
        ok = (idx >= 0) & nav.passable & nav.passable[np.maximum(idx, 0)]
        if not np.any(ok):
            continue
        dz = np.abs(nav.z[ok] - nav.z[idx[ok]])
        d = nav.neighbour_dist[ok, k]
        slope_ok = np.degrees(np.arctan2(dz, np.maximum(d, 1e-3))) <= vehicle.max_slope_deg
        # A link that spans several tiles bridges ground the sensor never
        # sampled, and a curb crossed obliquely over 3 m looks like a 2 deg
        # gradient.  Connectivity is therefore only asserted across links
        # short enough to be backed by observation; longer gaps leave the
        # far side *unknown* rather than silently drivable.
        adjacent = d <= max_link * nav.cell + 1e-6
        crossable = (dz <= vehicle.max_step_height) & slope_ok & adjacent
        rows.append(np.flatnonzero(ok)[crossable])
        cols.append(idx[ok][crossable])
    if not rows:
        return np.zeros(m, dtype=bool)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    labels = _components(m, rows, cols)
    d = np.hypot(nav.cx - seed_xy[0], nav.cy - seed_xy[1])
    cand = np.flatnonzero(nav.passable)
    if cand.size == 0:
        return np.zeros(m, dtype=bool)

    # Seeding from the single nearest passable tile is brittle: one stray
    # patch under the bumper and the whole road is declared unreachable.
    # Take instead the component that dominates the vehicle's immediate
    # surroundings - on a road that is the road.
    near = cand[d[cand] <= seed_radius]
    if near.size:
        ids, counts = np.unique(labels[near], return_counts=True)
        seed_label = ids[np.argmax(counts)]
    else:
        seed_label = labels[cand[np.argmin(d[cand])]]
    out = (labels == seed_label) & nav.passable
    nav.reachable = out
    return out


def _components(n: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if _cc is not None and coo_matrix is not None:
        g = coo_matrix((np.ones(rows.shape[0], np.int8), (rows, cols)), shape=(n, n))
        _, labels = _cc(g, directed=False)
        return labels
    labels = np.arange(n, dtype=np.int64)
    for _ in range(8192):
        upd = labels.copy()
        np.minimum.at(upd, rows, labels[cols])
        np.minimum.at(upd, cols, labels[rows])
        upd = np.minimum(upd, upd[np.clip(upd, 0, n - 1)])
        if np.array_equal(upd, labels):
            break
        labels = upd
    return labels
