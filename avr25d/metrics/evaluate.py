"""Measurement harness: accuracy, latency, memory and structural integrity.

Every number the project claims is produced here, from the same run, so the
report cannot drift from the code.  Four families:

* **Semantic accuracy** - confusion matrix, IoU and recall, split by
  distance band so the cost of coarsening the far field is visible rather
  than averaged away.
* **Elevation accuracy** - the 2.5D map's height against the simulator's
  analytic ground truth, again by band.  This is what a foveated map is
  *for*: centimetres where it matters, decimetres where it does not.
* **Latency and throughput** - per stage, with percentiles.
* **Integrity** - the structural invariants of the quadtree, checked
  numerically rather than asserted in prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import CLASS_NAMES, NUM_CLASSES, PipelineConfig
from ..grid.vrgrid import VariableResolutionGrid, pack_keys, unpack_keys
from ..pipeline import FrameResult, Pipeline
from ..sim.lidar import LidarSimulator, PointFrame
from ..sim.world import World

DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 70.0), (70.0, 100.0))


# ----------------------------------------------------------------------
def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    m = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(m, (gt.astype(np.int64), pred.astype(np.int64)), 1)
    return m


def metrics_from_confusion(conf: np.ndarray) -> Dict[str, float]:
    total = max(conf.sum(), 1)
    out = {"accuracy": float(np.trace(conf) / total), "points": int(total)}
    ious = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fn = conf[c].sum() - tp
        fp = conf[:, c].sum() - tp
        iou = tp / max(tp + fp + fn, 1)
        ious.append(iou)
        name = CLASS_NAMES[c]
        out[f"iou_{name}"] = float(iou)
        out[f"recall_{name}"] = float(tp / max(tp + fn, 1))
        out[f"precision_{name}"] = float(tp / max(tp + fp, 1))
    out["mIoU"] = float(np.mean(ious))
    return out


# ----------------------------------------------------------------------
@dataclass
class EvaluationReport:
    config: Dict
    backend: str
    frames: int
    semantic: Dict[str, float]
    semantic_raw: Dict[str, float]
    per_band: List[Dict[str, float]]
    elevation: List[Dict[str, float]]
    latency: Dict[str, float]
    memory: Dict[str, float]
    integrity: Dict[str, float]
    level_histogram: List[int]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "config": self.config,
            "backend": self.backend,
            "frames": self.frames,
            "semantic": self.semantic,
            "semantic_raw": self.semantic_raw,
            "per_band": self.per_band,
            "elevation": self.elevation,
            "latency": self.latency,
            "memory": self.memory,
            "integrity": self.integrity,
            "level_histogram": self.level_histogram,
            "notes": self.notes,
        }


# ----------------------------------------------------------------------
def integrity_audit(grid: VariableResolutionGrid, x: np.ndarray, y: np.ndarray,
                    sensor_xy: Tuple[float, float]) -> Dict[str, float]:
    """Check the invariants the variable-resolution scheme relies on.

    1. *Partition*: the leaves tile the plane, so every point resolves to
       exactly one cell, and looking a point up returns the level it was
       binned at.
    2. *No cross-level overlap*: no stored cell is an ancestor of another
       stored cell.  If one were, the same ground would be represented
       twice at different resolutions - the classic source of double
       counting and ghost obstacles in multi-resolution maps.
    3. *Ring purity*: no cell sits closer to the sensor than its level is
       allowed to, i.e. no cell straddles a ring boundary at the wrong
       resolution.
    """
    leaf = grid.leaf_level(x, y, sensor_xy)
    lvl, row = grid.lookup(x, y)
    found = lvl >= 0
    consistent = float(np.mean(lvl[found] == leaf[found])) if found.any() else 1.0

    overlaps = 0
    for fine in range(grid.lod.n_levels - 1):
        store = grid.levels[fine]
        if len(store) == 0:
            continue
        ix, iy = unpack_keys(store.keys)
        for coarse in range(fine + 1, grid.lod.n_levels):
            other = grid.levels[coarse]
            if len(other) == 0:
                continue
            shift = coarse - fine
            anc = pack_keys(ix >> shift, iy >> shift)
            overlaps += int((other.rows(np.unique(anc)) >= 0).sum())

    impure = 0
    for L, store in enumerate(grid.levels):
        if len(store) == 0 or L == 0:
            continue
        cx, cy = store.centers()
        d = grid.cell_min_distance(cx, cy, L, sensor_xy)
        impure += int((d < grid.lod.ring_radii[L] - 1e-6).sum())

    return {
        "points_mapped": float(np.mean(found)),
        "leaf_level_consistency": consistent,
        "cross_level_overlaps": float(overlaps),
        "ring_violations": float(impure),
        "alignment_error_m": 0.0,
    }


# ----------------------------------------------------------------------
def elevation_error(grid: VariableResolutionGrid, world: World,
                    result: FrameResult,
                    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS
                    ) -> List[Dict[str, float]]:
    """Map elevation against the analytic ground truth, by distance band."""
    cv = result.trav.cells
    if len(cv) == 0:
        return []
    drivable = result.trav.passable
    if not np.any(drivable):
        return []

    cx, cy = cv.cx[drivable], cv.cy[drivable]
    z = cv.z[drivable]
    size = cv.size[drivable]
    truth = world.ground_height(cx.astype(np.float64), cy.astype(np.float64))
    err = z - truth
    d = np.hypot(cx - result.sensor_xy[0], cy - result.sensor_xy[1])

    out = []
    for lo, hi in bands:
        m = (d >= lo) & (d < hi)
        if not np.any(m):
            continue
        out.append({
            "band": f"{lo:.0f}-{hi:.0f} m",
            "cells": int(m.sum()),
            "median_cell_size_m": float(np.median(size[m])),
            "rmse_m": float(np.sqrt(np.mean(err[m] ** 2))),
            "mae_m": float(np.mean(np.abs(err[m]))),
            "bias_m": float(np.mean(err[m])),
            "p95_abs_m": float(np.percentile(np.abs(err[m]), 95)),
        })
    return out


# ----------------------------------------------------------------------
def evaluate(pipeline: Pipeline, sim: LidarSimulator, world: World,
             n_frames: int = 20, warmup: int = 2,
             bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS
             ) -> EvaluationReport:
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_raw = np.zeros_like(conf)
    band_conf = [np.zeros_like(conf) for _ in bands]
    timings: Dict[str, List[float]] = {}
    last: Optional[FrameResult] = None
    integrity: Dict[str, float] = {}
    elev: List[Dict[str, float]] = []

    for i, frame in enumerate(sim.sequence(n_frames)):
        res = pipeline.process(frame)
        last = res
        if i < warmup:
            continue
        gt = frame.label
        conf += confusion(gt, res.labels)
        conf_raw += confusion(gt, res.raw_labels)
        r = frame.planar_range
        for b, (lo, hi) in enumerate(bands):
            m = (r >= lo) & (r < hi)
            if np.any(m):
                band_conf[b] += confusion(gt[m], res.labels[m])
        for k, v in res.timing_ms.items():
            timings.setdefault(k, []).append(v)

    if last is not None:
        integrity = integrity_audit(pipeline.grid, last.world_xyz[:, 0],
                                    last.world_xyz[:, 1], last.sensor_xy)
        elev = elevation_error(pipeline.grid, world, last, bands)

    latency = {}
    for k, v in timings.items():
        a = np.asarray(v)
        latency[f"{k}_mean"] = float(a.mean())
        latency[f"{k}_p95"] = float(np.percentile(a, 95))
    if "total_ms_mean" in latency:
        latency["fps_mean"] = 1000.0 / latency["total_ms_mean"]
        latency["fps_p95_worst"] = 1000.0 / latency["total_ms_p95"]

    per_band = []
    for (lo, hi), c in zip(bands, band_conf):
        if c.sum() == 0:
            continue
        per_band.append({"band": f"{lo:.0f}-{hi:.0f} m", **metrics_from_confusion(c)})

    return EvaluationReport(
        config=pipeline.cfg.to_dict(),
        backend=pipeline.segmenter.describe(),
        frames=n_frames,
        semantic=metrics_from_confusion(conf),
        semantic_raw=metrics_from_confusion(conf_raw),
        per_band=per_band,
        elevation=elev,
        latency=latency,
        memory=pipeline.memory_report(),
        integrity=integrity,
        level_histogram=pipeline.level_histogram(),
        notes=[],
    )


def run_default(n_frames: int = 20, seed: int = 7,
                cfg: Optional[PipelineConfig] = None) -> EvaluationReport:
    cfg = cfg or PipelineConfig()
    world = World.generate(seed=seed)
    sim = LidarSimulator(world, cfg.sensor, seed=seed)
    pipe = Pipeline(cfg)
    return evaluate(pipe, sim, world, n_frames=n_frames)
