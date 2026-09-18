"""2.5D A* over the navigation layer.

The planner is the honest test of a traversability map: if the map is wrong,
the path drives over a curb.  Cost is arc length scaled by how unpleasant
the destination tile is, plus an explicit climb penalty, so the planner
prefers smooth road, tolerates mild roughness, and refuses steps it cannot
climb - which is a 2.5D decision, not a 2D one.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import VehicleConfig
from ..grid.navigation import NavLayer


@dataclass
class Path:
    xy: np.ndarray                 # (K, 2)
    z: np.ndarray                  # (K,)
    cost: float
    expanded: int
    found: bool

    def __len__(self) -> int:
        return int(self.xy.shape[0])

    def length(self) -> float:
        if len(self) < 2:
            return 0.0
        d = np.diff(self.xy, axis=0)
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    def climb(self) -> float:
        return float(np.abs(np.diff(self.z)).sum()) if len(self) > 1 else 0.0


def _nearest(nav: NavLayer, xy: Tuple[float, float],
             mask: np.ndarray) -> Optional[int]:
    cand = np.flatnonzero(mask)
    if cand.size == 0:
        return None
    d = (nav.cx[cand] - xy[0]) ** 2 + (nav.cy[cand] - xy[1]) ** 2
    return int(cand[np.argmin(d)])


def plan(nav: NavLayer, start_xy: Tuple[float, float],
         goal_xy: Tuple[float, float],
         vehicle: Optional[VehicleConfig] = None,
         roughness_weight: float = 2.5,
         climb_weight: float = 6.0) -> Path:
    v = vehicle or VehicleConfig()
    if nav.neighbours is None:
        raise ValueError("navigation layer has not been linked")

    usable = nav.passable if nav.reachable is None else (nav.passable & nav.reachable)
    start = _nearest(nav, start_xy, usable)
    goal = _nearest(nav, goal_xy, usable)
    empty = Path(np.zeros((0, 2)), np.zeros(0), float("inf"), 0, False)
    if start is None or goal is None:
        return empty

    gx, gy = float(nav.cx[goal]), float(nav.cy[goal])
    quality = 1.0 - np.clip(nav.slope_deg / max(v.max_slope_deg, 1e-3), 0.0, 1.0)

    g_score = np.full(len(nav), np.inf, dtype=np.float64)
    came = np.full(len(nav), -1, dtype=np.int64)
    g_score[start] = 0.0
    open_heap = [(0.0, start)]
    closed = np.zeros(len(nav), dtype=bool)
    expanded = 0

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue
        closed[cur] = True
        expanded += 1
        if cur == goal:
            break
        for k in range(4):
            nxt = int(nav.neighbours[cur, k])
            if nxt < 0 or closed[nxt] or not usable[nxt]:
                continue
            dz = abs(float(nav.z[nxt]) - float(nav.z[cur]))
            if dz > v.max_step_height:
                continue
            d = float(nav.neighbour_dist[cur, k])
            cost = d * (1.0 + roughness_weight * (1.0 - float(quality[nxt]))) \
                + climb_weight * dz
            tentative = g_score[cur] + cost
            if tentative < g_score[nxt]:
                g_score[nxt] = tentative
                came[nxt] = cur
                h = np.hypot(nav.cx[nxt] - gx, nav.cy[nxt] - gy)
                heapq.heappush(open_heap, (tentative + float(h), nxt))

    if not np.isfinite(g_score[goal]):
        return Path(np.zeros((0, 2)), np.zeros(0), float("inf"), expanded, False)

    chain = [goal]
    while chain[-1] != start:
        chain.append(int(came[chain[-1]]))
    chain.reverse()
    idx = np.array(chain, dtype=np.int64)
    return Path(xy=np.stack([nav.cx[idx], nav.cy[idx]], axis=1),
                z=nav.z[idx], cost=float(g_score[goal]),
                expanded=expanded, found=True)
