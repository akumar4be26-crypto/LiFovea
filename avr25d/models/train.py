"""Train a segmentation backbone on simulated scans.

The simulator gives exact per-point labels, so supervision is free and the
whole training set is generated on the fly - no dataset download, no manual
annotation.  Worlds are split by seed, so the validation scenes contain
buildings, trees and traffic the network has never seen.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from ..config import CLASS_NAMES, NUM_CLASSES, PipelineConfig
from ..sim.lidar import LidarSimulator
from ..sim.world import World
from .features import compute_features, normalise
from . import nets
from .data import load_npz_dataset

try:
    import torch
    import torch.nn as nn
except Exception as exc:                            # pragma: no cover
    raise SystemExit("training requires PyTorch: pip install torch") from exc


# ----------------------------------------------------------------------
def make_dataset(seeds: List[int], frames_per_world: int = 3,
                 cfg: PipelineConfig = None) -> List[Dict[str, np.ndarray]]:
    cfg = cfg or PipelineConfig()
    out = []
    for s in seeds:
        world = World.generate(seed=s)
        sim = LidarSimulator(world, cfg.sensor, seed=s)
        for k in range(frames_per_world):
            frame = sim.scan(0.7 * k + 0.3 * s, frame_index=k)
            bundle = compute_features(frame.xyz, frame.intensity)
            out.append({
                "feats": normalise(bundle.features),
                "xyz": frame.xyz.astype(np.float32),
                "label": frame.label.astype(np.int64),
            })
    return out


def class_weights(dataset) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for s in dataset:
        known = s["label"] != 255
        counts += np.bincount(s["label"][known], minlength=NUM_CLASSES)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    return (w / w.mean()).astype(np.float32)


# ----------------------------------------------------------------------
def evaluate(net, dataset, device: str = "cpu") -> Dict[str, float]:
    net.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for s in dataset:
            logits = net(torch.from_numpy(s["feats"]).to(device),
                         torch.from_numpy(s["xyz"]).to(device))
            pred = logits.argmax(1).cpu().numpy()
            known = s["label"] != 255
            np.add.at(conf, (s["label"][known], pred[known]), 1)
    net.train()

    acc = float(np.trace(conf) / max(conf.sum(), 1))
    ious, recalls = [], []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fn = conf[c].sum() - tp
        fp = conf[:, c].sum() - tp
        ious.append(tp / max(tp + fp + fn, 1))
        recalls.append(tp / max(tp + fn, 1))
    return {
        "accuracy": acc,
        "mIoU": float(np.mean(ious)),
        **{f"iou_{CLASS_NAMES[c]}": float(ious[c]) for c in range(NUM_CLASSES)},
        **{f"recall_{CLASS_NAMES[c]}": float(recalls[c]) for c in range(NUM_CLASSES)},
    }


def train(architecture: str = "pointnet2lite", epochs: int = 12,
          train_seeds: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
          val_seeds: Tuple[int, ...] = (101, 102),
          frames_per_world: int = 3, subsample: int = 30000,
          lr: float = 2e-3, out: str = "checkpoints/avr25d.pt",
          device: str = "cpu", quiet: bool = False,
          real_data: str = None, real_val_data: str = None) -> Dict[str, float]:
    torch.manual_seed(0)
    np.random.seed(0)

    t0 = time.perf_counter()
    train_set = make_dataset(list(train_seeds), frames_per_world)
    val_set = make_dataset(list(val_seeds), max(1, frames_per_world - 1))
    if real_data:
        real_train = load_npz_dataset(real_data)
        train_set.extend(real_train)
        if real_val_data:
            val_set = load_npz_dataset(real_val_data)
    if not quiet:
        print(f"[data] {len(train_set)} train / {len(val_set)} val scans "
              f"in {time.perf_counter() - t0:.1f}s")

    net = nets.build(architecture).to(device)
    w = torch.from_numpy(class_weights(train_set)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=255)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(train_set))

    rng = np.random.default_rng(0)
    best = -1.0
    history = []
    for ep in range(epochs):
        order = rng.permutation(len(train_set))
        running = 0.0
        for i in order:
            s = train_set[i]
            known = np.flatnonzero(s["label"] != 255)
            if known.size == 0:
                continue
            if subsample and known.size > subsample:
                sel = rng.choice(known, subsample, replace=False)
            else:
                sel = known
            feats = torch.from_numpy(s["feats"][sel]).to(device)
            xyz = torch.from_numpy(s["xyz"][sel]).to(device)
            target = torch.from_numpy(s["label"][sel]).to(device)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(feats, xyz), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            running += float(loss)

        stats = evaluate(net, val_set, device)
        history.append({"epoch": ep, "loss": running / len(train_set), **stats})
        if not quiet:
            print(f"[{ep + 1:>2}/{epochs}] loss {running / len(train_set):.4f}  "
                  f"val acc {stats['accuracy']:.4f}  mIoU {stats['mIoU']:.4f}")
        if stats["mIoU"] > best:
            best = stats["mIoU"]
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            torch.save({"architecture": architecture,
                        "state_dict": net.state_dict(),
                        "val": stats}, out)

    final = evaluate(net, val_set, device)
    final["best_mIoU"] = best
    final["train_seconds"] = time.perf_counter() - t0
    final["checkpoint"] = out
    with open(os.path.splitext(out)[0] + "_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    return final


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="train the AVR-2.5D segmenter")
    ap.add_argument("--architecture", default="pointnet2lite",
                    choices=sorted(nets.ARCHITECTURES))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--frames-per-world", type=int, default=3)
    ap.add_argument("--subsample", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="checkpoints/avr25d.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real-data", help=".npz scan or directory for mixed training")
    ap.add_argument("--real-val-data", help="held-out .npz scan directory for validation")
    args = ap.parse_args(argv)
    stats = train(architecture=args.architecture, epochs=args.epochs,
                  frames_per_world=args.frames_per_world,
                  subsample=args.subsample, lr=args.lr, out=args.out,
                  device=args.device, real_data=args.real_data,
                  real_val_data=args.real_val_data)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
