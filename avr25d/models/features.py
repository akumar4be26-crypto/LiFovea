"""Shared per-point geometric feature stage.

Both the neural backends and the NumPy fallback classifier consume the same
feature tensor, which means the fallback is a genuine degraded mode of the
same pipeline rather than a separate code path with its own behaviour.

Everything here is O(N) with NumPy only: points are hashed into voxels and
columns, and per-group statistics are accumulated with ``bincount`` /
``reduceat`` instead of a neighbour search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

FEATURE_NAMES: Tuple[str, ...] = (
    "z",                 # height in the sensor frame
    "log_range",         # log(1 + planar range)
    "intensity",
    "height_above_low",  # height above the lowest return in the column
    "column_span",       # vertical extent of the column
    "log_density",       # log(1 + points in the voxel)
    "linearity",         # (l0 - l1) / l0   - poles, wires
    "planarity",         # (l1 - l2) / l0   - road, walls
    "scattering",        # l2 / l0          - foliage, pedestrians
    "verticality",       # 1 - |n_z|        - walls vs ground
    "normal_z",          # |n_z|
    "roughness",         # residual std about the local plane
    "elevation_angle",   # atan2(z, planar range)
)

NUM_FEATURES = len(FEATURE_NAMES)

_MASK = (1 << 21) - 1


def hash_keys(idx: np.ndarray) -> np.ndarray:
    """Pack integer voxel indices into a single int64 key."""
    a = (idx[:, 0].astype(np.int64) + (1 << 20)) & _MASK
    b = (idx[:, 1].astype(np.int64) + (1 << 20)) & _MASK
    if idx.shape[1] == 3:
        c = (idx[:, 2].astype(np.int64) + (1 << 20)) & _MASK
        return (a << 42) | (b << 21) | c
    return (a << 21) | b


def group(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique_keys, inverse, counts)."""
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return uniq, inv.astype(np.int64), cnt


def group_min_max(values: np.ndarray, inv: np.ndarray,
                  n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-group min and max without Python loops."""
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    sorted_val = values[order]
    starts = np.searchsorted(sorted_inv, np.arange(n_groups))
    gmin = np.minimum.reduceat(sorted_val, starts)
    gmax = np.maximum.reduceat(sorted_val, starts)
    # groups that are empty cannot happen (inv comes from np.unique)
    return gmin, gmax


@dataclass
class FeatureBundle:
    features: np.ndarray        # (N, NUM_FEATURES) float32
    voxel_inv: np.ndarray       # (N,) index into voxel groups
    voxel_count: np.ndarray     # (M,) points per voxel
    column_inv: np.ndarray      # (N,) index into 2D column groups
    column_min: np.ndarray      # (C,) lowest return per column
    column_max: np.ndarray
    ground_z: np.ndarray        # (N,) local ground elevation estimate


def compute_features(xyz: np.ndarray, intensity: np.ndarray,
                     voxel_size: float = 0.45,
                     column_size: float = 1.2) -> FeatureBundle:
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rng_xy = np.hypot(x, y)

    # ---- voxel grouping -------------------------------------------------
    vidx = np.floor(xyz / voxel_size).astype(np.int64)
    vkeys = hash_keys(vidx)
    _, vinv, vcnt = group(vkeys)
    m = vcnt.shape[0]

    # ---- per-voxel covariance via raw moments --------------------------
    w = np.ones(n, dtype=np.float64)
    s0 = np.bincount(vinv, weights=w, minlength=m)
    sx = np.bincount(vinv, weights=x, minlength=m)
    sy = np.bincount(vinv, weights=y, minlength=m)
    sz = np.bincount(vinv, weights=z, minlength=m)
    sxx = np.bincount(vinv, weights=x * x, minlength=m)
    syy = np.bincount(vinv, weights=y * y, minlength=m)
    szz = np.bincount(vinv, weights=z * z, minlength=m)
    sxy = np.bincount(vinv, weights=x * y, minlength=m)
    sxz = np.bincount(vinv, weights=x * z, minlength=m)
    syz = np.bincount(vinv, weights=y * z, minlength=m)

    inv_n = 1.0 / np.maximum(s0, 1.0)
    mx, my, mz = sx * inv_n, sy * inv_n, sz * inv_n
    cxx = sxx * inv_n - mx * mx
    cyy = syy * inv_n - my * my
    czz = szz * inv_n - mz * mz
    cxy = sxy * inv_n - mx * my
    cxz = sxz * inv_n - mx * mz
    cyz = syz * inv_n - my * mz

    cov = np.empty((m, 3, 3), dtype=np.float64)
    cov[:, 0, 0] = cxx; cov[:, 1, 1] = cyy; cov[:, 2, 2] = czz
    cov[:, 0, 1] = cov[:, 1, 0] = cxy
    cov[:, 0, 2] = cov[:, 2, 0] = cxz
    cov[:, 1, 2] = cov[:, 2, 1] = cyz
    cov[:, 0, 0] += 1e-9
    cov[:, 1, 1] += 1e-9
    cov[:, 2, 2] += 1e-9

    evals, evecs = np.linalg.eigh(cov)          # ascending
    l2, l1, l0 = evals[:, 0], evals[:, 1], evals[:, 2]
    l0 = np.maximum(l0, 1e-12)
    linearity = (l0 - l1) / l0
    planarity = (l1 - l2) / l0
    scattering = np.clip(l2 / l0, 0.0, 1.0)
    normal = evecs[:, :, 0]                     # eigenvector of smallest eval
    normal_z = np.abs(normal[:, 2])
    roughness = np.sqrt(np.maximum(l2, 0.0))

    # single-point voxels carry no shape information
    thin = s0 < 4
    linearity[thin] = 0.0
    planarity[thin] = 0.0
    scattering[thin] = 1.0
    normal_z[thin] = 0.0
    roughness[thin] = 0.0

    # ---- column (2D) grouping ------------------------------------------
    cidx = np.floor(xyz[:, :2] / column_size).astype(np.int64)
    ckeys = hash_keys(cidx)
    _, cinv, _ = group(ckeys)
    c = int(cinv.max()) + 1
    cmin, cmax = group_min_max(z.astype(np.float64), cinv, c)

    ground_z = cmin[cinv]
    height_above_low = z - ground_z
    column_span = (cmax - cmin)[cinv]

    feats = np.empty((n, NUM_FEATURES), dtype=np.float32)
    feats[:, 0] = z
    feats[:, 1] = np.log1p(rng_xy)
    feats[:, 2] = intensity
    feats[:, 3] = height_above_low
    feats[:, 4] = column_span
    feats[:, 5] = np.log1p(vcnt)[vinv]
    feats[:, 6] = linearity[vinv]
    feats[:, 7] = planarity[vinv]
    feats[:, 8] = scattering[vinv]
    feats[:, 9] = 1.0 - normal_z[vinv]
    feats[:, 10] = normal_z[vinv]
    feats[:, 11] = roughness[vinv]
    feats[:, 12] = np.arctan2(z, np.maximum(rng_xy, 1e-3))

    return FeatureBundle(
        features=feats,
        voxel_inv=vinv,
        voxel_count=vcnt,
        column_inv=cinv,
        column_min=cmin,
        column_max=cmax,
        ground_z=ground_z.astype(np.float32),
    )


#: per-feature normalisation used before any network sees the tensor
FEATURE_MEAN = np.array(
    [-0.9, 2.7, 0.35, 0.55, 1.6, 2.0, 0.35, 0.45, 0.18, 0.35, 0.65, 0.07, -0.20],
    dtype=np.float32)
FEATURE_STD = np.array(
    [1.4, 1.0, 0.25, 1.1, 2.2, 1.2, 0.28, 0.28, 0.20, 0.32, 0.32, 0.09, 0.28],
    dtype=np.float32)


def normalise(features: np.ndarray) -> np.ndarray:
    return (features - FEATURE_MEAN) / FEATURE_STD
