"""Build the demo payload the dashboard runs on.

Runs the real pipeline over a short sequence, keeps the classified point
clouds (quantised to centimetres), and bundles them with the benchmark
report.  The dashboard re-derives the variable-resolution grid from these
points in the browser, which is what lets the ring sliders work live
instead of showing pre-baked pictures.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from avr25d.config import CLASS_COLORS, CLASS_NAMES, PipelineConfig
from avr25d.grid.vrgrid import BYTES_PER_CELL
from avr25d.metrics.evaluate import evaluate
from avr25d.pipeline import Pipeline
from avr25d.sim.lidar import LidarSimulator
from avr25d.sim.world import World


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _city_objects(world: World, timestamp: float) -> List[Dict]:
    objects = []
    for primitive in world.objects_at(timestamp):
        objects.append({
            "kind": int(primitive.kind),
            "params": [round(float(v), 3) for v in primitive.params],
            "label": int(primitive.label),
        })
    return objects


def build(n_frames: int = 6, seed: int = 7, keep_every: int = 1,
        benchmark_frames: int = 12, checkpoint: str = None) -> Dict:
    cfg = PipelineConfig()
    if checkpoint:
      cfg.model.checkpoint = checkpoint
      cfg.model.backend = "torch"
      cfg.validate()
    world = World.generate(seed=seed, traffic=28, pedestrians=40)
    sim = LidarSimulator(world, cfg.sensor, seed=seed)
    pipe = Pipeline(cfg)

    frames: List[Dict] = []
    for frame in sim.sequence(n_frames):
        res = pipe.process(frame)
        rel = frame.xyz.astype(np.float64)
        if keep_every > 1:
            sel = slice(None, None, keep_every)
            rel, labels = rel[sel], res.labels[sel]
        else:
            labels = res.labels
        frames.append({
            "index": int(frame.frame_index),
            "t": float(frame.timestamp),
            "sensor": [float(v) for v in frame.origin],
            "yaw": float(frame.yaw),
            "n": int(rel.shape[0]),
            "x": _b64(np.round(rel[:, 0] * 100).astype(np.int16)),
            "y": _b64(np.round(rel[:, 1] * 100).astype(np.int16)),
            "z": _b64(np.round(rel[:, 2] * 100).astype(np.int16)),
            "label": _b64(labels.astype(np.int8)),
            "timing": {k: round(v, 2) for k, v in res.timing_ms.items()},
            "cells": int(res.memory["cells"]),
            "city": _city_objects(world, frame.timestamp),
        })

    bench_world = World.generate(seed=seed, traffic=28, pedestrians=40)
    bench_sim = LidarSimulator(bench_world, cfg.sensor, seed=seed)
    report = evaluate(Pipeline(cfg), bench_sim, bench_world,
                      n_frames=benchmark_frames)

    return {
        "meta": {
            "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "backend": pipe.segmenter.describe(),
            "checkpoint": Path(cfg.model.checkpoint).name if checkpoint else "default",
            "base_cell": cfg.lod.base_cell,
            "n_levels": cfg.lod.n_levels,
            "ring_radii": list(cfg.lod.ring_radii),
            "max_range": cfg.lod.max_range,
            "nav_cell": cfg.vehicle.nav_cell,
            "max_slope_deg": cfg.vehicle.max_slope_deg,
            "max_step_height": cfg.vehicle.max_step_height,
            "clearance_height": cfg.vehicle.clearance_height,
            "near_zone_radius": cfg.model.foveal_radius,
            "transition_zone_radius": cfg.model.transition_radius,
            "transition_stride": cfg.model.transition_stride,
            "far_stride": cfg.model.far_stride,
            "bytes_per_cell": BYTES_PER_CELL,
            "class_names": list(CLASS_NAMES),
            "class_colors": [list(c) for c in CLASS_COLORS],
            "sensor": {"beams": cfg.sensor.n_beams,
                       "azimuth_steps": cfg.sensor.azimuth_steps,
                       "rpm": cfg.sensor.rpm},
        },
        "benchmark": report.to_dict(),
        "frames": frames,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--benchmark-frames", type=int, default=12)
    ap.add_argument("--keep-every", type=int, default=1)
    ap.add_argument("--out", default="web/demo_data.json")
    args = ap.parse_args(argv)

    payload = build(n_frames=args.frames, keep_every=args.keep_every,
                    benchmark_frames=args.benchmark_frames)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}  ({size / 1e6:.2f} MB, {len(payload['frames'])} frames)")


if __name__ == "__main__":
    main()
