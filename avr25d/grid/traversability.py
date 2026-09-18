"""Traversability analysis over the variable-resolution 2.5D grid.

Three questions are answered per cell:

* **Is the surface itself drivable?**  slope, step to the neighbours, and
  in-cell relief, all measured against the platform's limits.
* **Is anything in the way?**  obstacle and dynamic-object evidence, plus
  the overhead-clearance layer - a branch at 4 m is not an obstacle, which
  is precisely the distinction a flat 2D occupancy grid destroys.
* **Can the vehicle get there?**  reachability from the vehicle's own cell
  across passable edges.  Flat tarmac on the far side of a 16 cm curb is
  not drivable, and only a connectivity test says so.

Neighbour queries go through :meth:`VariableResolutionGrid.lookup`, which
returns the finest cell covering a position, so a 5 cm cell and the 40 cm
cell next to it are neighbours without any special-case code.
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
    VehicleConfig,
)
from . import navigation
from .vrgrid import CellView, VariableResolutionGrid

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
except Exception:                                   # pragma: no cover
    coo_matrix = None
    _cc = None

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class TraversabilityResult:
    cells: CellView
    slope_deg: np.ndarray
    step: np.ndarray
    score: np.ndarray          # 0 = impassable, 1 = clear
    passable: np.ndarray       # bool, local test only
    reachable: np.ndarray      # bool, connected to the vehicle
    terrain: np.ndarray        # CLASS_DRIVABLE / CLASS_ROUGH_TERRAIN
    known: np.ndarray          # connectivity is supported by observation here
    neighbours: np.ndarray     # (N, 4) global cell index, -1 where absent
    neighbour_dist: np.ndarray  # (N, 4) separation in metres
    nav: object = None         # the navigation layer used for reachability

    def __len__(self) -> int:
        return int(self.score.shape[0])


def _global_offsets(grid: VariableResolutionGrid) -> np.ndarray:
    counts = [len(s) for s in grid.levels]
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def neighbour_index(grid: VariableResolutionGrid, cv: CellView,
                    max_probe: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """First occupied cell in each of the four directions.

    A single revolution samples the ground far more densely in azimuth than
    in range, so strict 4-adjacency leaves the fine cells as isolated
    islands and any connectivity test collapses.  Probing outwards until the
    first occupied cell is found - at most ``max_probe`` cell widths - walks
    over the radial sampling gap without inventing surface that was never
    observed.  The distance actually travelled is returned so slope is
    measured over the true separation.
    """
    off = _global_offsets(grid)
    n = len(cv)
    out = np.full((n, 4), -1, dtype=np.int64)
    dist = np.full((n, 4), np.inf, dtype=np.float32)
    self_idx = np.arange(n, dtype=np.int64)

    for k, (dx, dy) in enumerate(_DIRS):
        todo = np.ones(n, dtype=bool)
        for step in range(1, max_probe + 1):
            idx = np.flatnonzero(todo)
            if idx.size == 0:
                break
            off_m = step * cv.size[idx]
            qx = cv.cx[idx] + dx * off_m
            qy = cv.cy[idx] + dy * off_m
            lvl, row = grid.lookup(qx, qy)
            hit = lvl >= 0
            if np.any(hit):
                gidx = off[lvl[hit].astype(np.int64)] + row[hit]
                # a coarse neighbour can be the cell itself: keep probing
                real = gidx != self_idx[idx[hit]]
                tgt = idx[hit][real]
                out[tgt, k] = gidx[real]
                dist[tgt, k] = off_m[hit][real]
                todo[tgt] = False
    return out, dist


def analyse(grid: VariableResolutionGrid,
            vehicle: Optional[VehicleConfig] = None,
            sensor_xy: Tuple[float, float] = (0.0, 0.0),
            nav_cell: Optional[float] = None,
            reach_horizon: float = 45.0) -> TraversabilityResult:
    v = vehicle or VehicleConfig()
    nav_cell = v.nav_cell if nav_cell is None else nav_cell
    cv = grid.cells()
    n = len(cv)
    if n == 0:
        z = np.zeros(0)
        return TraversabilityResult(cv, z, z, z, z.astype(bool), z.astype(bool),
                                    z.astype(np.int8), z.astype(bool),
                                    np.zeros((0, 4), np.int64),
                                    np.zeros((0, 4), np.float32))

    relief = np.maximum(cv.z_max - cv.z_min, 0.0)

    # ---- semantic gating ---------------------------------------------
    blocked_by_class = (cv.cls == CLASS_STATIC_OBSTACLE) | (cv.cls == CLASS_DYNAMIC_OBJECT)

    # Overhead clearance: an obstacle return that passes *over* the vehicle
    # does not block the cell.  A branch at 4 m and a wall at 4 m look
    # identical to a 2D occupancy grid; they do not look identical here.
    surface_ok = (relief <= v.max_step_height) & (cv.n_obs > 0)

    # ---- geometry and reachability on the navigation layer ------------
    # Slope, step and connectivity are relations *between* patches of
    # ground, and at 5 cm a single revolution does not even give adjacent
    # patches - the returns are islands.  So those questions are asked at
    # vehicle scale on the navigation layer and the answers come back down.
    nav = navigation.build(cv, surface_ok, blocked_by_class, nav_cell=nav_cell,
                           clearance=v.clearance_height)
    navigation.link(nav, v)
    reach_nav = navigation.reachability(nav, v, sensor_xy)

    parent = np.maximum(nav.parent, 0)
    blocked = blocked_by_class & nav.blocked[parent]
    # Where the ground is sampled densely enough for connectivity to mean
    # something, "not reached" is a verdict; past that horizon it is only an
    # absence of evidence, and the map says so instead of guessing.
    d_sensor = np.hypot(cv.cx - sensor_xy[0], cv.cy - sensor_xy[1])
    known = d_sensor <= reach_horizon
    slope = nav.slope_deg[parent].astype(np.float32)
    step = nav.step[parent].astype(np.float32)
    passable = surface_ok & ~blocked & (slope <= v.max_slope_deg)
    reachable = reach_nav[parent] & passable

    # ---- continuous score --------------------------------------------
    pen_slope = slope / max(v.max_slope_deg, 1e-3)
    pen_step = step / max(v.max_step_height, 1e-3)
    pen_relief = relief / max(v.max_roughness * 2.5, 1e-3)
    score = np.clip(1.0 - np.maximum.reduce([pen_slope, pen_step, pen_relief]),
                    0.0, 1.0).astype(np.float32)
    score[blocked] = 0.0
    score[~reachable] = np.minimum(score[~reachable], 0.15)

    terrain = np.where(reachable & passable, CLASS_DRIVABLE,
                       CLASS_ROUGH_TERRAIN).astype(np.int8)
    terrain[blocked] = np.where(cv.cls[blocked] == CLASS_DYNAMIC_OBJECT,
                                CLASS_DYNAMIC_OBJECT, CLASS_STATIC_OBSTACLE)

    return TraversabilityResult(cells=cv, slope_deg=slope, step=step, score=score,
                                passable=passable, reachable=reachable,
                                terrain=terrain, known=known,
                                neighbours=np.zeros((0, 4), np.int64),
                                neighbour_dist=np.zeros((0, 4), np.float32),
                                nav=nav)


# ----------------------------------------------------------------------
def terrain_for_points(grid: VariableResolutionGrid,
                       result: TraversabilityResult,
                       x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Push the grid's terrain verdict back onto points.

    The point network answers 'is this the ground?'; the grid answers 'is
    that ground drivable?', because reachability is a property of the map,
    not of a single return.
    """
    off = _global_offsets(grid)
    lvl, row = grid.lookup(x, y)
    out = np.full(x.shape[0], CLASS_ROUGH_TERRAIN, dtype=np.int8)
    ok = lvl >= 0
    gidx = off[lvl[ok].astype(np.int64)] + row[ok]
    verdict = result.reachable[gidx] & result.passable[gidx]
    # outside the horizon the map has no opinion, so it does not overrule one
    verdict |= ~result.known[gidx]
    out[ok] = np.where(verdict, CLASS_DRIVABLE, CLASS_ROUGH_TERRAIN)
    return out
