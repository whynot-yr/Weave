"""Compatibility import; the canonical encoder lives outside the simulator package."""

from weave_policy.models.dinov2_encoder import DinoV2Encoder

__all__ = ["DinoV2Encoder"]
