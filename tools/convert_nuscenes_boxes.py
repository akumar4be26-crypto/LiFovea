"""Convert nuScenes keyframes to weakly labeled AVR-2.5D NPZ scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

UNKNOWN = 255
DYNAMIC_WORDS = ("vehicle", "pedestrian", "bicycle", "motorcycle", "cyclist")
STATIC_WORDS = ("barrier", "traffic_cone", "construction", "debris")


def quat_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def read_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, 5).astype(np.float64)


def convert(root: Path, output: Path, every: int = 1) -> int:
    table = root / "v1.0-mini"
    sample_data = json.loads((table / "sample_data.json").read_text())
    calibrated = {x["token"]: x for x in json.loads((table / "calibrated_sensor.json").read_text())}
    sensors = {x["token"]: x for x in json.loads((table / "sensor.json").read_text())}
    poses = {x["token"]: x for x in json.loads((table / "ego_pose.json").read_text())}
    categories = {x["token"]: x["name"] for x in json.loads((table / "category.json").read_text())}
    instances = {x["token"]: x["category_token"] for x in json.loads((table / "instance.json").read_text())}
    annotations = {}
    for row in json.loads((table / "sample_annotation.json").read_text()):
        annotations.setdefault(row["sample_token"], []).append(row)

    lidar = []
    for row in sample_data:
        cal = calibrated[row["calibrated_sensor_token"]]
        if sensors[cal["sensor_token"]]["channel"] == "LIDAR_TOP" and row["is_key_frame"]:
            lidar.append(row)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in lidar[::max(every, 1)]:
        cal = calibrated[row["calibrated_sensor_token"]]
        pose = poses[row["ego_pose_token"]]
        r_cal = quat_matrix(cal["rotation"])
        r_ego = quat_matrix(pose["rotation"])
        sensor_to_global = r_ego @ r_cal
        sensor_origin = np.asarray(pose["translation"]) + r_ego @ np.asarray(cal["translation"])
        points = read_points(root / row["filename"])
        xyz_global = points[:, :3] @ sensor_to_global.T + sensor_origin
        labels = np.full(points.shape[0], UNKNOWN, dtype=np.uint8)

        for ann in annotations.get(row["sample_token"], []):
            name = categories[instances[ann["instance_token"]]]
            cls = 3 if name.startswith(DYNAMIC_WORDS) else 2 if name.startswith(STATIC_WORDS) else None
            if cls is None:
                continue
            center = np.asarray(ann["translation"], dtype=np.float64)
            size = np.asarray(ann["size"], dtype=np.float64)
            r_box = quat_matrix(ann["rotation"])
            local = (xyz_global - center) @ r_box
            inside = np.all(np.abs(local) <= size / 2, axis=1)
            labels[inside] = cls

        np.savez_compressed(output / f"{row['token']}.npz",
                            xyz=points[:, :3].astype(np.float32),
                            intensity=points[:, 3].astype(np.float32),
                            label=labels)
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="data/real/nuscenes/extracted")
    parser.add_argument("--out", default="data/real/nuscenes/weak_npz")
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args(argv)
    count = convert(Path(args.root), Path(args.out), args.every)
    print(f"wrote {count} weakly labeled scans to {args.out}")


if __name__ == "__main__":
    main()
