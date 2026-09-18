"""Backend selection and point-wise inference.

``Segmenter`` is the single entry point the pipeline talks to.  It picks a
backend according to the config and what is actually installed, keeps the
timing, and always returns ``(labels, probabilities)`` whichever path ran -
so the rest of the system never branches on whether PyTorch is present.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np

from ..config import ModelConfig, NUM_CLASSES, VehicleConfig
from .features import FeatureBundle, compute_features, normalise
from .geometric import GeometricSegmenter

try:
    import torch
    _TORCH = True
except Exception:                                   # pragma: no cover
    torch = None
    _TORCH = False


def torch_available() -> bool:
    return _TORCH


class Segmenter:
    """Point-wise semantic segmentation with a graceful fallback."""

    def __init__(self, cfg: Optional[ModelConfig] = None,
                 vehicle: Optional[VehicleConfig] = None):
        self.cfg = cfg or ModelConfig()
        self.vehicle = vehicle or VehicleConfig()
        self.fallback = GeometricSegmenter(vehicle=self.vehicle)
        self.net = None
        self.backend = "numpy"
        self.last_timing = {}

        want_torch = self.cfg.backend in ("auto", "torch")
        if want_torch and _TORCH and self.cfg.architecture != "geometric":
            ckpt = self.cfg.checkpoint
            if ckpt and os.path.exists(ckpt):
                self._load(ckpt)
            elif self.cfg.backend == "torch":
                raise FileNotFoundError(
                    f"backend='torch' requires a checkpoint; {ckpt!r} not found. "
                    "Train one with `python -m avr25d.cli train`.")
        if self.cfg.backend == "torch" and self.net is None:
            raise RuntimeError("PyTorch backend requested but unavailable")

    # ------------------------------------------------------------------
    def _load(self, path: str) -> None:
        from . import nets
        blob = torch.load(path, map_location="cpu", weights_only=False)
        arch = blob.get("architecture", self.cfg.architecture)
        net = nets.build(arch)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        net.to(self.cfg.device)
        self.net = net
        self.backend = f"torch:{arch}"

    # ------------------------------------------------------------------
    def describe(self) -> str:
        if self.net is None:
            return "numpy:geometric"
        return self.backend

    # ------------------------------------------------------------------
    def predict(self, xyz: np.ndarray, intensity: np.ndarray,
                bundle: Optional[FeatureBundle] = None
                ) -> Tuple[np.ndarray, np.ndarray, FeatureBundle]:
        t0 = time.perf_counter()
        if bundle is None:
            bundle = compute_features(xyz, intensity)
        t1 = time.perf_counter()

        if self.net is None:
            labels, probs = self.fallback.predict(xyz, intensity, bundle)
        else:
            labels, probs = self._predict_torch(xyz, bundle)
        t2 = time.perf_counter()

        self.last_timing = {
            "features_ms": (t1 - t0) * 1e3,
            "inference_ms": (t2 - t1) * 1e3,
        }
        return labels, probs, bundle

    # ------------------------------------------------------------------
    def _select(self, xyz: np.ndarray, bundle: FeatureBundle) -> np.ndarray:
        """Select three foveated inference zones.

        Near points stay full 3D, the transition zone uses a lighter stride,
        and the far zone uses a coarser stride plus one representative per
        voxel so skipped points still receive semantic evidence.
        """
        transition_stride = max(int(self.cfg.transition_stride), 1)
        far_stride = max(int(self.cfg.far_stride), 1)
        if transition_stride == 1 and far_stride == 1:
            return np.arange(xyz.shape[0])
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        idx = np.arange(xyz.shape[0])
        near = r < self.cfg.foveal_radius
        transition = (r >= self.cfg.foveal_radius) & (r < self.cfg.transition_radius)
        far = r >= self.cfg.transition_radius
        keep_transition = idx[transition][::transition_stride]
        keep_far = idx[far][::far_stride]
        _, first = np.unique(bundle.voxel_inv, return_index=True)
        return np.unique(np.concatenate([idx[near], keep_transition, keep_far, first]))

    def _predict_torch(self, xyz: np.ndarray, bundle: FeatureBundle
                       ) -> Tuple[np.ndarray, np.ndarray]:
        feats = normalise(bundle.features)
        sel = self._select(xyz, bundle)
        with torch.inference_mode():
            f = torch.from_numpy(np.ascontiguousarray(feats[sel])).float()
            p = torch.from_numpy(np.ascontiguousarray(xyz[sel])).float()
            logits = self.net(f.to(self.cfg.device), p.to(self.cfg.device))
            probs_sel = torch.softmax(logits, dim=1).cpu().numpy()

        if sel.shape[0] == xyz.shape[0]:
            probs = probs_sel
        else:
            # share each voxel's evidence with the points that were skipped
            inv = bundle.voxel_inv
            m = int(inv.max()) + 1
            acc = np.zeros((m, probs_sel.shape[1]), dtype=np.float32)
            for c in range(probs_sel.shape[1]):
                acc[:, c] = np.bincount(inv[sel], weights=probs_sel[:, c],
                                        minlength=m)
            tot = np.maximum(acc.sum(axis=1, keepdims=True), 1e-6)
            probs = (acc / tot)[inv]
            probs[sel] = probs_sel
        return probs.argmax(axis=1).astype(np.int8), probs.astype(np.float32)
