"""Depth preprocessing and the four-convolution student encoder (no simulator imports)."""

from __future__ import annotations

import torch
from torch import nn

DEPTH_SIZE = 128
DEPTH_NEAR = 0.1
DEPTH_FAR = 10.0


def depth_to_mm(depth_m: torch.Tensor) -> torch.Tensor:
    """Encode image-plane depth in meters; zero denotes invalid, output is uint16."""
    valid = torch.isfinite(depth_m) & (depth_m >= DEPTH_NEAR) & (depth_m <= DEPTH_FAR)
    return torch.where(valid, torch.round(depth_m * 1000.0), 0).to(torch.uint16)


def depth_from_mm(depth_mm: torch.Tensor) -> torch.Tensor:
    """Decode to the fixed CNN input range; identical for replay and online inference."""
    return depth_mm.to(torch.float32) / (1000.0 * DEPTH_FAR)


def preprocess_depth(depth_m: torch.Tensor) -> torch.Tensor:
    """Quantize both online and replay depth identically, without per-image normalization."""
    return depth_from_mm(depth_to_mm(depth_m))


class DepthCNNEncoder(nn.Module):
    """Four stride-two convolutions followed by a spatially preserving latent projection."""

    def __init__(self, latent_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip((1, 32, 64, 128), (32, 64, 128, 128)):
            layers.extend((nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1), nn.SiLU()))
        self.convs = nn.Sequential(*layers)
        self.projection = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convs(depth).flatten(1))
