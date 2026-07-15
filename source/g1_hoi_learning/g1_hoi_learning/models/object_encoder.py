"""PointNet++ object point-cloud encoder for observations.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .pointnet2_rl_seg import PointNet2RLSegmenter

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "pointnet2_rl_seg.pth")


class ObjectPointCloudEncoder(nn.Module):
    """Frozen seg-backbone encoder: ``(B, P, 3) -> (B, output_dim)``.
    """

    def __init__(self, ckpt_path: str = _DEFAULT_WEIGHTS, output_dim: int = 128, normalize: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.normalize = normalize

        self._seg = PointNet2RLSegmenter(input_channels=0, output_dim=output_dim)
        state = torch.load(ckpt_path, map_location="cpu")
        self._seg.load_state_dict(state)  # strict: full segmenter (encoder + FP + head)
        self._seg.eval()
        for p in self._seg.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: D401 - keep frozen encoder in eval mode always
        return super().train(False)

    @torch.no_grad()
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, P, 3) surface points in any metric frame.
        if self.normalize:
            centroid = pts.mean(dim=1, keepdim=True)                 # (B, 1, 3)
            pts = pts - centroid
            scale = pts.norm(dim=-1).amax(dim=1).clamp_min(1e-6)     # (B,)
            pts = pts / scale[:, None, None]
        feats = self._seg.extract_point_features(pts)               # (B, P, 128)
        return feats.mean(dim=1)                                     # (B, output_dim)


_ENCODERS: dict[torch.device, ObjectPointCloudEncoder] = {}


def get_object_encoder(device: torch.device) -> ObjectPointCloudEncoder:
    """Lazily build and cache the encoder per device."""
    enc = _ENCODERS.get(device)
    if enc is None:
        enc = ObjectPointCloudEncoder().to(device)
        enc.eval()
        _ENCODERS[device] = enc
    return enc
