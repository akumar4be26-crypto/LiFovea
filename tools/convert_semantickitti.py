"""Convert SemanticKITTI sequences into AVR-2.5D labeled NPZ scans."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# SemanticKITTI semantic ids grouped into the project's four classes.
DRIVABLE = {40, 44, 48, 49, 60}
ROUGH = {0, 1, 52}
STATIC = {10, 13, 15, 16, 18, 20, 30, 31, 32, 81}
DYNAMIC = {11, 12, 13, 15, 16, 17, 18, 19, 20, 21}


def map_labels(raw: np.ndarray) -> np.ndarray:
    semantic = raw.astype(np.uint32) & 0xFFFF
    labels = np.full(semantic.shape, 1, dtype=np.int8)
    labels[np.isin(semantic, list(DRIVABLE))] = 0
    labels[np.isin(semantic, list(STATIC))] = 2
    labels[np.isin(semantic, list(DYNAMIC))] = 3
    labels[np.isin(semantic, list(ROUGH))] = 1
    return labels


def convert_sequence(sequence: Path, output: Path, every: int = 1) -> int:
    scans = sequence / "velodyne"
    labels = sequence / "labels"
    files = sorted(scans.glob("*.bin"))[::max(every, 1)]
    if not files:
        raise FileNotFoundError(f"no velodyne/*.bin scans under {scans}")
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for scan in files:
        label_file = labels / f"{scan.stem}.label"
        if not label_file.exists():
            raise FileNotFoundError(f"missing label file for {scan.name}: {label_file}")
        points = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)
        raw_labels = np.fromfile(label_file, dtype=np.uint32)
        if len(points) != len(raw_labels):
            raise ValueError(f"{scan}: point/label count mismatch")
        out_file = output / f"{scan.stem}.npz"
        np.savez_compressed(
            out_file,
            xyz=points[:, :3],
            intensity=points[:, 3],
            label=map_labels(raw_labels),
        )
        written += 1
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="SemanticKITTI root containing sequences/")
    parser.add_argument("--sequences", nargs="+", required=True,
                        help="sequence ids, e.g. 00 01 02")
    parser.add_argument("--out", default="data/real/train")
    parser.add_argument("--every", type=int, default=1,
                        help="keep every Nth scan to control dataset size")
    args = parser.parse_args(argv)
    root = Path(args.root)
    total = 0
    for sequence_id in args.sequences:
        sequence = root / "sequences" / sequence_id.zfill(2)
        count = convert_sequence(sequence, Path(args.out) / sequence_id.zfill(2), args.every)
        total += count
        print(f"sequence {sequence_id}: {count} scans")
    print(f"wrote {total} scans to {args.out}")


if __name__ == "__main__":
    main()
