"""Single-critic WEAVE policy; legacy class name retained for configuration compatibility."""
import torch
from .networks import SimBaActorCritic


class MultiCriticSimBaActorCritic(SimBaActorCritic):
    def update_normalization(self, obs):
        # RSL-RL collects under inference_mode; its normalizer replaces _std.
        # Keep persistent buffers normal tensors so checkpoint reload is legal.
        with torch.inference_mode(False), torch.no_grad():
            super().update_normalization(obs)
