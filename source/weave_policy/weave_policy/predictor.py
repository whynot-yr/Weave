"""Checkpoint inference from cached DINO tokens or live RGB; no robot state input."""

import torch

from .flow_matching_policy import FlowMatchingPolicy
from .models.dinov2_encoder import DinoV2Encoder


class ReferencePredictor:
    def __init__(self, checkpoint, device="cpu", *, encoder=None):
        self.policy, self.metadata = FlowMatchingPolicy.from_checkpoint(checkpoint, device)
        self.encoder = None
        if encoder is not None:
            self.set_encoder(encoder)

    def set_encoder(self, encoder):
        metadata = encoder.metadata()
        if metadata["encoder_id"] != self.metadata["encoder"]["encoder_id"]:
            raise ValueError("DINOv2 weights/preprocessing differ from training")
        self.encoder = encoder.to(self.policy.device).eval()

    def load_encoder(self, repo_dir=None, weights=None, repo_ref="main"):
        self.set_encoder(DinoV2Encoder.from_pretrained(repo_dir, weights, repo_ref))

    @torch.no_grad()
    def predict_tokens(self, tokens, encoder_id, *, seed=None, num_steps=None):
        if encoder_id != self.metadata["encoder"]["encoder_id"]:
            raise ValueError("Cached token encoder differs from training")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.policy.device).manual_seed(seed)
        return self.policy.sample(tokens, num_steps=num_steps, generator=generator).cpu()

    @torch.no_grad()
    def predict_rgb(self, rgb, *, seed=None, num_steps=None):
        if self.encoder is None:
            raise RuntimeError("Load the matching DINOv2 encoder before live RGB inference")
        tokens = self.encoder(rgb)
        return self.predict_tokens(tokens, self.metadata["encoder"]["encoder_id"], seed=seed, num_steps=num_steps)
