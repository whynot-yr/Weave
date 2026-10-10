"""GR00T DiT time conditioning, matching diffusers Timesteps/TimestepEmbedding.

This differs from action-time sin encoding: 256 channels, denominator 127,
cos-before-sin, followed by Linear -> SiLU -> Linear.
"""

import math

import torch
from torch import nn


class Timesteps(nn.Module):
    def forward(self, timesteps):
        frequency = torch.exp(-math.log(10000.0) * torch.arange(128, device=timesteps.device).float() / 127)
        angles = timesteps[:, None].float() * frequency[None]
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear_1 = nn.Linear(256, width)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(width, width)

    def forward(self, value):
        return self.linear_2(self.act(self.linear_1(value)))


class TimestepEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.time_proj = Timesteps()
        self.timestep_embedder = TimestepEmbedding(width)

    def forward(self, timestep):
        dtype = self.timestep_embedder.linear_1.weight.dtype
        return self.timestep_embedder(self.time_proj(timestep).to(dtype))
