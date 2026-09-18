"""Command line interface.

    python -m avr25d.cli demo      run the pipeline over a few frames
    python -m avr25d.cli bench     measure accuracy / latency / memory
    python -m avr25d.cli train     train a segmentation backbone
    python -m avr25d.cli export    build the dashboard payload
    python -m avr25d.cli serve     run the dashboard locally
    python -m avr25d.cli plan      plan a route on the current map
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

from .config import CLASS_NAMES, PipelineConfig
from .metrics.evaluate import evaluate
from .pipeline import Pipeline
from .sim.lidar import LidarSimulator
from .sim.world import World


def _stack(cfg: PipelineConfig, seed: int):
    world = World.generate(seed=seed)
    sim = LidarSimulator(world, cfg.sensor, seed=seed)
    return world, sim, Pipeline(cfg)


def _config_from_args(args) -> PipelineConfig:
    cfg = PipelineConfig()
    if getattr(args, "backend", None):
        cfg.model.backend = args.backend
    if getattr(args, "checkpoint", None):
        cfg.model.checkpoint = args.checkpoint
    if getattr(args, "base_cell", None):
        cfg.lod.base_cell = args.base_cell
    if getattr(args, "rings", None):
        cfg.lod.ring_radii = tuple(float(v) for v in args.rings.split(","))
        cfg.lod.n_levels = len(cfg.lod.ring_radii)
    return cfg.validate()


# ----------------------------------------------------------------------
def cmd_demo(args) -> int:
    cfg = _config_from_args(args)
    world, sim, pipe = _stack(cfg, args.seed)
    print(f"[*] backend      : {pipe.segmenter.describe()}")
    print(f"[*] cell sizes   : "
          f"{', '.join(f'{c * 100:.0f} cm' for c in cfg.lod.cell_sizes)}")
    print(f"[*] ring radii   : {', '.join(f'{r:.0f} m' for r in cfg.lod.ring_radii)}")
    print()
    hdr = f"{'frame':>5} {'points':>8} {'acc':>7} {'ms':>7} {'fps':>6} {'cells':>9} {'map':>9}"
    print(hdr)
    print("-" * len(hdr))
    accs = []
    for frame in sim.sequence(args.frames):
        res = pipe.process(frame)
        acc = float((res.labels == frame.label).mean())
        accs.append(acc)
        print(f"{frame.frame_index:>5} {len(frame):>8,} {acc:>7.3f} "
              f"{res.timing_ms['total_ms']:>7.1f} {res.fps:>6.1f} "
              f"{int(res.memory['cells']):>9,} "
              f"{res.memory['adaptive_bytes'] / 1e6:>7.2f} MB")
    mem = pipe.memory_report()
    print()
    print(f"[+] mean point accuracy      : {np.mean(accs):.3f}")
    print(f"[+] cells per level          : {pipe.level_histogram()}")
    print(f"[+] adaptive map             : {mem['adaptive_bytes'] / 1e6:.2f} MB")
    print(f"[+] uniform {cfg.lod.base_cell * 100:.0f} cm grid would be : "
          f"{mem['uniform_fine_bytes'] / 1e6:.1f} MB "
          f"({mem['reduction_vs_uniform_fine']:.0f}x larger)")
    print(f"[+] dense 3D voxel volume    : {mem['dense_voxel_bytes'] / 1e6:.1f} MB "
          f"({mem['reduction_vs_dense_voxel']:.0f}x larger)")
    return 0


# ----------------------------------------------------------------------
def _markdown(report) -> str:
    d = report.to_dict()
    s = d["semantic"]
    lines = [
        "# AVR-2.5D benchmark", "",
        f"- backend: `{d['backend']}`",
        f"- frames: {d['frames']}",
        f"- point accuracy: **{s['accuracy']:.3f}**, mIoU **{s['mIoU']:.3f}**",
        f"- network-only accuracy: {d['semantic_raw']['accuracy']:.3f} "
        f"(mIoU {d['semantic_raw']['mIoU']:.3f})",
        f"- end-to-end latency: {d['latency']['total_ms_mean']:.1f} ms "
        f"({d['latency']['fps_mean']:.1f} fps)",
        f"- map: {d['memory']['adaptive_bytes'] / 1e6:.2f} MB, "
        f"{d['memory']['reduction_vs_uniform_fine']:.0f}x smaller than a uniform "
        f"{report.config['lod']['base_cell'] * 100:.0f} cm grid",
        "", "## Per class", "",
        "| class | IoU | recall | precision |", "|---|---:|---:|---:|",
    ]
    for c in CLASS_NAMES:
        lines.append(f"| {c} | {s['iou_' + c]:.3f} | {s['recall_' + c]:.3f} "
                     f"| {s['precision_' + c]:.3f} |")
    lines += ["", "## By range", "", "| band | points | accuracy | mIoU |",
              "|---|---:|---:|---:|"]
    for b in d["per_band"]:
        lines.append(f"| {b['band']} | {b['points']:,} | {b['accuracy']:.3f} "
                     f"| {b['mIoU']:.3f} |")
    lines += ["", "## Elevation error vs ground truth", "",
              "| band | median cell | RMSE | P95 abs | cells |",
              "|---|---:|---:|---:|---:|"]
    for e in d["elevation"]:
        lines.append(f"| {e['band']} | {e['median_cell_size_m'] * 100:.0f} cm "
                     f"| {e['rmse_m'] * 1000:.0f} mm | {e['p95_abs_m'] * 1000:.0f} mm "
                     f"| {e['cells']:,} |")
    lines += ["", "## Latency", "", "| stage | mean ms | p95 ms |", "|---|---:|---:|"]
    for k in ("features", "inference", "projection", "traversability", "total"):
        mk, pk = f"{k}_ms_mean", f"{k}_ms_p95"
        if mk in d["latency"]:
            lines.append(f"| {k} | {d['latency'][mk]:.1f} | {d['latency'][pk]:.1f} |")
    lines += ["", "## Structural integrity", "", "| invariant | value |", "|---|---:|"]
    for k, v in d["integrity"].items():
        lines.append(f"| {k} | {v:g} |")
    return "\n".join(lines) + "\n"


def cmd_bench(args) -> int:
    cfg = _config_from_args(args)
    world, sim, pipe = _stack(cfg, args.seed)
    report = evaluate(pipe, sim, world, n_frames=args.frames)
    d = report.to_dict()
    print(f"backend        {d['backend']}")
    print(f"accuracy       {d['semantic']['accuracy']:.4f}   "
          f"mIoU {d['semantic']['mIoU']:.4f}")
    print(f"network only   {d['semantic_raw']['accuracy']:.4f}   "
          f"mIoU {d['semantic_raw']['mIoU']:.4f}")
    print(f"latency        {d['latency']['total_ms_mean']:.1f} ms  "
          f"({d['latency']['fps_mean']:.1f} fps)")
    print(f"memory         {d['memory']['adaptive_bytes'] / 1e6:.2f} MB  "
          f"({d['memory']['reduction_vs_uniform_fine']:.0f}x smaller)")
    print(f"integrity      {d['integrity']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(d, fh, indent=2)
        print(f"[+] wrote {args.json}")
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(_markdown(report))
        print(f"[+] wrote {args.report}")
    return 0


# ----------------------------------------------------------------------
def cmd_plan(args) -> int:
    from .planner import astar
    cfg = _config_from_args(args)
    world, sim, pipe = _stack(cfg, args.seed)
    res = None
    for frame in sim.sequence(args.frames):
        res = pipe.process(frame)
    nav = res.trav.nav
    sx, sy = res.sensor_xy
    goal = (sx + args.ahead, sy + args.lateral)
    path = astar.plan(nav, (sx, sy), goal, cfg.vehicle)
    if not path.found:
        print(f"[!] no route to ({goal[0]:.1f}, {goal[1]:.1f}) - "
              "the goal is not connected to the vehicle across passable terrain")
        return 1
    print(f"[+] route found: {len(path)} tiles, {path.length():.1f} m travelled, "
          f"{path.climb():.2f} m climbed, {path.expanded} nodes expanded")
    for i in range(0, len(path), max(1, len(path) // 12)):
        print(f"    {path.xy[i, 0] - sx:>7.1f} {path.xy[i, 1] - sy:>7.1f}  "
              f"z={path.z[i]:.2f}")
    return 0


# ----------------------------------------------------------------------
def cmd_train(args) -> int:
    from .models.train import main as train_main
    argv = ["--architecture", args.architecture, "--epochs", str(args.epochs),
            "--out", args.out]
    train_main(argv)
    return 0


def cmd_export(args) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.export_demo import main as export_main
    export_main(["--frames", str(args.frames), "--out", args.out,
                 "--benchmark-frames", str(args.benchmark_frames)])
    return 0


def cmd_serve(args) -> int:
    from .server.app import serve
    serve(host=args.host, port=args.port, frames=args.frames,
          open_browser=args.open, checkpoint=args.checkpoint)
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="avr25d", description="Adaptive variable-resolution 2.5D lidar mapping")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--backend", choices=["auto", "torch", "numpy"])
        p.add_argument("--checkpoint")
        p.add_argument("--base-cell", type=float, dest="base_cell")
        p.add_argument("--rings", help="comma separated ring radii, e.g. 0,10,20,40,70")

    p = sub.add_parser("demo", help="run the pipeline and print per-frame stats")
    p.add_argument("--frames", type=int, default=10)
    common(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("bench", help="full measurement run")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--json", help="write the raw report here")
    p.add_argument("--report", help="write a markdown report here")
    common(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("plan", help="plan a route over the current map")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--ahead", type=float, default=35.0)
    p.add_argument("--lateral", type=float, default=0.0)
    common(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("train", help="train a segmentation backbone")
    p.add_argument("--architecture", default="pointnet2lite")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--out", default="checkpoints/avr25d.pt")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="build the dashboard payload")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--benchmark-frames", type=int, default=12)
    p.add_argument("--out", default="web/demo_data.json")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("serve", help="run the dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--open", action="store_true", help="open a browser window")
    p.add_argument("--checkpoint", help="serve using a specific ML checkpoint")
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
