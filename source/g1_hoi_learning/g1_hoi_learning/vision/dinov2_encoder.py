"""Frozen DINOv2 ViT-B/14 encoder returning CLS and every spatial patch token.

Input: RGB uint8 [B, 224, 224, 3]. Output: [B, 257, 768], with the CLS token
first and the 16x16 patch grid in the model's original order. This module does
not depend on Isaac Sim and can be used for both offline extraction and rollout.
"""

from pathlib import Path

import torch
from torch import nn


class DinoV2Encoder(nn.Module):
    IMAGE_SIZE = 224
    PATCH_COUNT = 256
    FEATURE_DIM = 768

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if getattr(backbone, "num_register_tokens", 0) != 0:
            raise ValueError("Expected DINOv2 ViT-B/14 without register tokens")
        if getattr(backbone, "embed_dim", self.FEATURE_DIM) != self.FEATURE_DIM:
            raise ValueError("Expected a 768-dimensional DINOv2 ViT-B/14 backbone")
        if getattr(backbone, "patch_size", 14) != 14:
            raise ValueError("Expected patch size 14")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.eval()

    @classmethod
    def from_pretrained(cls, repo_dir: str | Path | None = None) -> "DinoV2Encoder":
        """Load official pretrained weights via PyTorch Hub.

        Pass a locally checked-out facebookresearch/dinov2 directory to avoid
        fetching repository code at runtime. PyTorch Hub may still download the
        pretrained weights on first use unless they are already cached.
        """
        if repo_dir is None:
            backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        else:
            backbone = torch.hub.load(str(Path(repo_dir).expanduser()), "dinov2_vitb14", source="local")
        return cls(backbone)

    def train(self, mode: bool = True) -> "DinoV2Encoder":
        """Keep the frozen backbone in evaluation mode when its parent trains."""
        super().train(False)
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if not isinstance(rgb, torch.Tensor) or rgb.dtype != torch.uint8:
            raise TypeError("Expected a uint8 RGB tensor")
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (self.IMAGE_SIZE, self.IMAGE_SIZE, 3):
            raise ValueError("Expected RGB images shaped [B, 224, 224, 3]")

        image = rgb.to(device=self.image_mean.device).permute(0, 3, 1, 2).float().div(255.0)
        image = (image - self.image_mean) / self.image_std
        with torch.no_grad():
            features = self.backbone.forward_features(image)
            cls_token = features["x_norm_clstoken"]
            patch_tokens = features["x_norm_patchtokens"]
            if cls_token.shape != (len(rgb), self.FEATURE_DIM) or patch_tokens.shape != (
                len(rgb),
                self.PATCH_COUNT,
                self.FEATURE_DIM,
            ):
                raise ValueError("DINOv2 returned an unexpected token shape")
            return torch.cat((cls_token.unsqueeze(1), patch_tokens), dim=1)
