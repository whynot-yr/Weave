"""Contract checks without downloading DINOv2 weights or starting Isaac Sim."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

MODULE = Path(__file__).resolve().parents[1] / "g1_hoi_learning/vision/dinov2_encoder.py"
spec = importlib.util.spec_from_file_location("dinov2_encoder", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DinoV2Encoder = module.DinoV2Encoder


class DummyBackbone(nn.Module):
    embed_dim = 768
    patch_size = 14
    num_register_tokens = 0

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.input = None

    def forward_features(self, image):
        self.input = image
        batch = len(image)
        return {
            "x_norm_clstoken": torch.ones(batch, 768, device=image.device),
            "x_norm_patchtokens": torch.full((batch, 256, 768), 2.0, device=image.device),
        }


def test_token_order_preprocessing_and_frozen_mode():
    backbone = DummyBackbone()
    encoder = DinoV2Encoder(backbone)
    encoder.train()
    rgb = torch.zeros(2, 224, 224, 3, dtype=torch.uint8)
    tokens = encoder(rgb)

    assert tokens.shape == (2, 257, 768)
    assert torch.all(tokens[:, 0] == 1)
    assert torch.all(tokens[:, 1:] == 2)
    assert torch.allclose(backbone.input[0, :, 0, 0], -encoder.image_mean[0, :, 0, 0] / encoder.image_std[0, :, 0, 0])
    assert not backbone.training
    assert not backbone.weight.requires_grad


def test_rejects_wrong_image_contract():
    encoder = DinoV2Encoder(DummyBackbone())
    with pytest.raises(TypeError, match="uint8"):
        encoder(torch.zeros(1, 224, 224, 3))
    with pytest.raises(ValueError, match="224"):
        encoder(torch.zeros(1, 256, 256, 3, dtype=torch.uint8))
