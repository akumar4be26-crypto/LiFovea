"""Inspect extracted nuScenes mini lidar and object annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_pcd(path: Path) -> np.ndarray:
    """Read nuScenes' binary PCD payload as float32 x/y/z/intensity points."""
    if path.name.endswith(".pcd.bin"):
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 5:
            raise ValueError(f"unexpected nuScenes binary point size: {path}")
        return points.reshape(-1, 5)
    header = bytearray()
    with path.open("rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"missing PCD DATA header: {path}")
            header.extend(line)
            if line.strip().lower() == b"data binary":
                break
        points = np.frombuffer(fh.read(), dtype=np.float32)
    if points.size % 5:
        raise ValueError(f"unexpected nuScenes point record size: {path}")
    return points.reshape(-1, 5)


def inspect(root: Path, limit: int = 20) -> Dict:
    table = root / "v1.0-mini"
    sample_data = json.loads((table / "sample_data.json").read_text())
    calibrated = json.loads((table / "calibrated_sensor.json").read_text())
    sensors = json.loads((table / "sensor.json").read_text())
    annotations = json.loads((table / "sample_annotation.json").read_text())
    samples = json.loads((table / "sample.json").read_text())
    sample_by_token = {row["token"]: row for row in samples}
    sensor_by_token = {row["token"]: row for row in sensors}
    calibrated_by_token = {row["token"]: row for row in calibrated}
    lidar = []
    for row in sample_data:
        calibration = calibrated_by_token[row["calibrated_sensor_token"]]
        sensor = sensor_by_token[calibration["sensor_token"]]
        if sensor["channel"] == "LIDAR_TOP":
            lidar.append(row)
    lidar_by_sample = {row["sample_token"]: row for row in lidar}

    counts: List[int] = []
    for row in lidar[:limit]:
        points = read_pcd(root / row["filename"])
        counts.append(int(points.shape[0]))

    annotations_by_sample: Dict[str, int] = {}
    for row in annotations:
        annotations_by_sample[row["sample_token"]] = annotations_by_sample.get(row["sample_token"], 0) + 1

    return {
        "lidar_keyframes": len(lidar),
        "sample_annotations": len(annotations),
        "scans_checked": len(counts),
        "points_min": min(counts) if counts else 0,
        "points_mean": float(np.mean(counts)) if counts else 0.0,
        "points_max": max(counts) if counts else 0,
        "annotated_samples": sum(1 for token in lidar_by_sample if token in annotations_by_sample),
        "sample_scenes": len({sample_by_token[row["sample_token"]]["scene_token"] for row in lidar}),
        "point_level_labels": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/real/nuscenes/extracted")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    report = inspect(Path(args.root), args.limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
