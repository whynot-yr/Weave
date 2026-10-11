"""Checkpointed architecture and temporal protocol."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReferencePolicyConfig:
    reference_dim: int = 125
    horizon: int = 5
    reference_offsets: tuple[int, ...] = (0, 5, 10, 15, 20)
    fps: float = 50.0
    width: int = 768
    image_dim: int = 768
    image_tokens: int = 257
    num_layers: int = 16
    num_heads: int = 16
    head_dim: int = 48
    dropout: float = 0.2
    num_embodiments: int = 1
    num_timestep_buckets: int = 1000
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_inference_steps: int = 4
    normalization: str = "minmax"
    clip_outliers: bool = True
    protocol: str = "weave-reference-flow-v1"

    def __post_init__(self):
        if self.reference_dim != 125 or self.horizon != 5:
            raise ValueError("The reference protocol requires 5 tokens of 125 dimensions")
        if len(self.reference_offsets) != self.horizon or self.reference_offsets[0] != 0:
            raise ValueError("Expected 5 reference offsets starting at zero")
        if any(type(x) is not int or x < 0 for x in self.reference_offsets):
            raise ValueError("Reference offsets must be nonnegative integers")
        if any(a >= b for a, b in zip(self.reference_offsets, self.reference_offsets[1:])):
            raise ValueError("Reference offsets must be strictly increasing")
        if self.image_dim != 768 or self.image_tokens != 257:
            raise ValueError("Expected CLS + 256 DINOv2 ViT-B/14 patch tokens")
        if self.width != self.num_heads * self.head_dim or self.width % 2:
            raise ValueError("Width must be even and equal to heads * head_dim")
        if min(self.width, self.num_layers, self.num_embodiments, self.num_inference_steps) <= 0:
            raise ValueError("Architecture dimensions and inference steps must be positive")
        if self.num_timestep_buckets <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid time buckets or dropout")
        if any(not math.isfinite(x) or x <= 0 for x in (self.fps, self.noise_beta_alpha, self.noise_beta_beta)):
            raise ValueError("FPS and Beta parameters must be finite and positive")
        if not 0 < self.noise_s < 1:
            raise ValueError("noise_s must lie in (0,1)")
        if self.normalization not in {"minmax", "minmax_extrema", "meanstd"}:
            raise ValueError("Unknown normalization configuration")
        if self.protocol != "weave-reference-flow-v1":
            raise ValueError("Unknown reference protocol")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if "reference_offsets" in value:
            value["reference_offsets"] = tuple(value["reference_offsets"])
        return cls(**value)
