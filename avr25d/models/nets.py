"""PyTorch backbones for point-wise semantic segmentation.

Two architectures, both hierarchical and both sparse:

``PointNet2Lite``
    A PointNet++ in the shape that matters - shared point-wise MLPs,
    symmetric max-pooling inside local neighbourhoods, several scales, and
    feature propagation back to every point.  Neighbourhoods come from a
    voxel hash rather than farthest-point sampling plus ball query: same
    grouping semantics, deterministic, and O(N) instead of O(N log N) with
    a kernel this project would have to ship itself.

``SparseVoxelNet``
    The sparse-convolution alternative.  Points are pooled into pillars,
    the occupied pillars are scattered into a bird's-eye feature map, a
    small 2D CNN with dilations gives every pillar a wide receptive field,
    and the result is gathered back onto the points.  This is the
    architecture that sees *context* - a flat patch surrounded by road is
    road; the same patch surrounded by wall is a pavement.

Both consume the shared feature tensor from :mod:`avr25d.models.features`,
so swapping backends changes nothing upstream or downstream.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES
from .features import NUM_FEATURES


def _mlp(sizes, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if last_act or i < len(sizes) - 2:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def scatter_max(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Max-pool ``src`` rows into ``n`` groups given by ``index``."""
    out = torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype)
    return out.index_reduce_(0, index, src, "amax", include_self=False)


def voxel_group(xyz: torch.Tensor, size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group indices and group centroids for a voxel grid of side ``size``."""
    idx = torch.floor(xyz / size).to(torch.int64)
    keys = (idx[:, 0] + (1 << 20)) * (1 << 42) + (idx[:, 1] + (1 << 20)) * (1 << 21)
    if idx.shape[1] == 3:
        keys = keys + (idx[:, 2] + (1 << 20))
    uniq, inv = torch.unique(keys, return_inverse=True)
    n = uniq.shape[0]
    cnt = torch.zeros(n, device=xyz.device).index_add_(
        0, inv, torch.ones(xyz.shape[0], device=xyz.device))
    cent = torch.zeros(n, xyz.shape[1], device=xyz.device).index_add_(0, inv, xyz)
    return inv, cent / cnt.clamp_min(1.0).unsqueeze(1)


# ----------------------------------------------------------------------
class PointNet2Lite(nn.Module):
    """Two-scale set abstraction with feature propagation."""

    name = "pointnet2lite"

    def __init__(self, in_dim: int = NUM_FEATURES, n_classes: int = NUM_CLASSES,
                 r1: float = 0.6, r2: float = 2.4, width: int = 64):
        super().__init__()
        self.r1, self.r2 = r1, r2
        self.point_mlp = _mlp([in_dim + 3, width, width])
        self.sa1 = _mlp([width + 3, width, 2 * width])
        self.sa2 = _mlp([2 * width + 3, 2 * width, 4 * width])
        # Feature propagation projects each scale down before concatenation.
        # Concatenating the raw scales would push ~700 channels through a
        # per-point layer, which is most of the inference budget for no
        # measurable accuracy.
        self.fp1 = nn.Linear(2 * width, width)
        self.fp2 = nn.Linear(4 * width, width)
        self.fpg = nn.Linear(4 * width, width)
        self.head = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.BatchNorm1d(2 * width), nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(2 * width, width), nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Linear(width, n_classes),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        inv1, cent1 = voxel_group(xyz, self.r1)
        local1 = xyz - cent1[inv1]
        p = self.point_mlp(torch.cat([feats, local1], dim=1))

        g1 = scatter_max(torch.cat([p, local1], dim=1), inv1, cent1.shape[0])
        g1 = self.sa1(g1)

        inv2, cent2 = voxel_group(cent1, self.r2)
        local2 = cent1 - cent2[inv2]
        g2 = scatter_max(torch.cat([g1, local2], dim=1), inv2, cent2.shape[0])
        g2 = self.sa2(g2)

        glob = g2.max(dim=0, keepdim=True).values.expand(p.shape[0], -1)

        # feature propagation: broadcast each level back to its members
        fused = torch.cat([p, self.fp1(g1)[inv1], self.fp2(g2)[inv2][inv1],
                           self.fpg(glob)], dim=1)
        return self.head(fused)


# ----------------------------------------------------------------------
class SparseVoxelNet(nn.Module):
    """Pillar encoder + dilated BEV CNN + per-point head."""

    name = "sparsevoxel"

    def __init__(self, in_dim: int = NUM_FEATURES, n_classes: int = NUM_CLASSES,
                 pillar: float = 0.5, extent: float = 100.0, width: int = 32):
        super().__init__()
        self.pillar = pillar
        self.extent = extent
        self.side = int(2 * extent / pillar)
        self.encoder = _mlp([in_dim + 3, width, width])
        self.cnn = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(True),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.BatchNorm2d(width),
            nn.ReLU(True),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.BatchNorm1d(2 * width),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(2 * width, n_classes),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        s, e = self.pillar, self.extent
        j = torch.clamp(((xyz[:, 0] + e) / s).long(), 0, self.side - 1)
        i = torch.clamp(((xyz[:, 1] + e) / s).long(), 0, self.side - 1)
        flat = i * self.side + j

        cx = (j.to(xyz.dtype) + 0.5) * s - e
        cy = (i.to(xyz.dtype) + 0.5) * s - e
        local = torch.stack([xyz[:, 0] - cx, xyz[:, 1] - cy, xyz[:, 2]], dim=1)
        p = self.encoder(torch.cat([feats, local], dim=1))

        bev = scatter_max(p, flat, self.side * self.side)
        bev = bev.view(1, self.side, self.side, -1).permute(0, 3, 1, 2).contiguous()
        bev = self.cnn(bev)
        bev = bev.permute(0, 2, 3, 1).reshape(self.side * self.side, -1)

        return self.head(torch.cat([p, bev[flat]], dim=1))


ARCHITECTURES = {
    PointNet2Lite.name: PointNet2Lite,
    SparseVoxelNet.name: SparseVoxelNet,
}


def build(architecture: str, **kwargs) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; "
                         f"choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[architecture](**kwargs)
