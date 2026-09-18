"""Dataset adapters for simulator and labeled real-world lidar scans."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .features import compute_features, normalise


def load_npz_dataset(path: str) -> List[Dict[str, np.ndarray]]:
    """Load labeled lidar scans from one ``.npz`` file or a directory.

    Each archive must contain ``xyz`` with shape ``(N, 3)`` and ``label`` with
    shape ``(N,)``. ``intensity`` is optional and defaults to ones. Labels use
    classes 0 through 3; 255 is accepted as unknown and ignored in training.
    """
    files = [path] if os.path.isfile(path) else [
        os.path.join(path, name) for name in sorted(os.listdir(path))
        if name.endswith(".npz")
    ]
    if not files:
        raise FileNotFoundError(f"no .npz lidar scans found at {path!r}")

    dataset: List[Dict[str, np.ndarray]] = []
    for filename in files:
        with np.load(filename) as archive:
            if "xyz" not in archive or "label" not in archive:
                raise ValueError(f"{filename!r} requires 'xyz' and 'label' arrays")
            xyz = np.asarray(archive["xyz"], dtype=np.float32)
            labels = np.asarray(archive["label"], dtype=np.int64)
            intensity = np.asarray(
                archive["intensity"] if "intensity" in archive
                else np.ones(xyz.shape[0], dtype=np.float32),
                dtype=np.float32,
            )
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"{filename!r}: xyz must have shape (N, 3)")
        if labels.shape != (xyz.shape[0],):
            raise ValueError(f"{filename!r}: label must have shape (N,)")
        if intensity.shape != (xyz.shape[0],):
            raise ValueError(f"{filename!r}: intensity must have shape (N,)")
        if np.any((labels != 255) & ((labels < 0) | (labels > 3))):
            raise ValueError(f"{filename!r}: labels must be integers in [0, 3] or 255")
        bundle = compute_features(xyz, intensity)
        dataset.append({
            "feats": normalise(bundle.features),
            "xyz": xyz,
            "label": labels,
        })
    return dataset
