"""End-to-end perception pipeline.

    scan -> features -> semantic segmentation -> variable-resolution
    projection -> Bayesian elevation fusion -> traversability -> map

One :class:`Pipeline` instance holds the persistent map, so consecutive
frames accumulate: cells fill in as the vehicle drives, and the quadtree
rebalances around the moving sensor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    CLASS_DRIVABLE,
    CLASS_DYNAMIC_OBJECT,
    CLASS_ROUGH_TERRAIN,
    CLASS_STATIC_OBSTACLE,
    PipelineConfig,
    uniform_cell_count,
)
from .grid import traversability as tv
from .grid.vrgrid import BYTES_PER_CELL, VariableResolutionGrid
from .models.infer import Segmenter
from .sim.lidar import PointFrame


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    sensor_xy: Tuple[float, float]
    sensor_z: float
    yaw: float
    n_points: int
    labels: np.ndarray                 # point labels after grid refinement
    raw_labels: np.ndarray             # point labels straight from the network
    world_xyz: np.ndarray
    trav: tv.TraversabilityResult
    timing_ms: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, float] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        total = self.timing_ms.get("total_ms", 0.0)
        return 1000.0 / total if total > 0 else float("nan")


class Pipeline:
    """Stateful perception stack over a stream of LiDAR frames."""

    def __init__(self, cfg: Optional[PipelineConfig] = None,
                 persistent: bool = True):
        self.cfg = (cfg or PipelineConfig()).validate()
        self.grid = VariableResolutionGrid(self.cfg)
        self.segmenter = Segmenter(self.cfg.model, self.cfg.vehicle)
        self.persistent = persistent
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    @staticmethod
    def to_world(frame: PointFrame) -> np.ndarray:
        c, s = np.cos(frame.yaw), np.sin(frame.yaw)
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return frame.xyz @ r.T + frame.origin

    # ------------------------------------------------------------------
    def process(self, frame: PointFrame) -> FrameResult:
        t_start = time.perf_counter()
        if not self.persistent:
            self.grid.reset()

        # ---- 1. semantics ---------------------------------------------
        labels, probs, bundle = self.segmenter.predict(frame.xyz, frame.intensity)
        t_seg = time.perf_counter()

        # ---- 2. project into the variable-resolution grid --------------
        world = self.to_world(frame)
        sensor_xy = (float(frame.origin[0]), float(frame.origin[1]))
        if self.persistent and self.grid.cell_count():
            self.grid.rebalance(sensor_xy)
        ground_mask = labels <= CLASS_ROUGH_TERRAIN
        self.grid.update(world[:, 0], world[:, 1], world[:, 2], probs,
                         frame.intensity, sensor_xy, float(frame.origin[2]),
                         ground_mask=ground_mask, frame_index=frame.frame_index)
        if self.persistent:
            self.grid.prune(sensor_xy)
        t_grid = time.perf_counter()

        # ---- 3. traversability + terrain refinement --------------------
        trav = tv.analyse(self.grid, self.cfg.vehicle, sensor_xy)
        refined = labels.copy()
        drivable = labels == CLASS_DRIVABLE
        if np.any(drivable):
            # The map may only ever *revoke* drivability, never grant it.
            # Reachability is information the point network cannot have -
            # tarmac behind a curb looks like tarmac - but the network sees
            # local geometry the 80 cm far-field cells have thrown away, so
            # letting the map promote points would trade real detail for a
            # coarse guess.  Demotion only: the conservative direction.
            verdict = tv.terrain_for_points(
                self.grid, trav, world[drivable, 0], world[drivable, 1])
            idx = np.flatnonzero(drivable)
            refined[idx[verdict != CLASS_DRIVABLE]] = CLASS_ROUGH_TERRAIN
        t_trav = time.perf_counter()

        timing = {
            "features_ms": self.segmenter.last_timing.get("features_ms", 0.0),
            "inference_ms": self.segmenter.last_timing.get("inference_ms", 0.0),
            "projection_ms": (t_grid - t_seg) * 1e3,
            "traversability_ms": (t_trav - t_grid) * 1e3,
            "total_ms": (t_trav - t_start) * 1e3,
        }
        memory = self.memory_report(frame)

        self.history.append({**timing, **memory,
                             "frame": float(frame.frame_index)})
        return FrameResult(
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            sensor_xy=sensor_xy,
            sensor_z=float(frame.origin[2]),
            yaw=frame.yaw,
            n_points=len(frame),
            labels=refined,
            raw_labels=labels,
            world_xyz=world,
            trav=trav,
            timing_ms=timing,
            memory=memory,
        )

    # ------------------------------------------------------------------
    def memory_report(self, frame: Optional[PointFrame] = None) -> Dict[str, float]:
        """Bytes held by the adaptive map against the obvious alternatives."""
        lod = self.cfg.lod
        adaptive = self.grid.memory_bytes()
        uniform_fine = uniform_cell_count(lod.max_range, lod.base_cell) * BYTES_PER_CELL
        uniform_coarse = uniform_cell_count(
            lod.max_range, lod.cell_size(lod.n_levels - 1)) * BYTES_PER_CELL

        # a dense 3D voxel volume at the same finest resolution, 20 m tall,
        # counting only the bytes of an occupancy bit per voxel
        side = int(2 * lod.max_range / lod.base_cell)
        vox = side * side * int(20.0 / lod.base_cell) / 8.0

        out = {
            "adaptive_bytes": float(adaptive),
            "uniform_fine_bytes": float(uniform_fine),
            "uniform_coarse_bytes": float(uniform_coarse),
            "dense_voxel_bytes": float(vox),
            "cells": float(self.grid.cell_count()),
            "reduction_vs_uniform_fine": float(uniform_fine / max(adaptive, 1)),
            "reduction_vs_dense_voxel": float(vox / max(adaptive, 1)),
        }
        if frame is not None:
            raw = float(frame.nbytes())
            out["raw_points_bytes"] = raw
            out["reduction_vs_raw_points"] = raw / max(adaptive, 1)
        return out

    # ------------------------------------------------------------------
    def level_histogram(self) -> List[int]:
        return [len(s) for s in self.grid.levels]

    def reset(self) -> None:
        self.grid.reset()
        self.history.clear()
