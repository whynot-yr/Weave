# Adapted from NVIDIA Isaac-GR00T embodiment_conditioned_mlp.py (Apache-2.0).
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T's action/time fusion, with reference125 and width 768."""

import math

import torch
from torch import nn


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, output_dim))

    def forward(self, value, category_ids):
        return torch.bmm(value, self.W[category_ids]) + self.b[category_ids].unsqueeze(1)


class SinusoidalPositionalEncoding(nn.Module):
    """GR00T action-time formula: sin then cos, denominator width/2."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half = self.embedding_dim // 2
        frequency = torch.exp(-torch.arange(half, device=timesteps.device).float() * math.log(10000.0) / half)
        angles = timesteps.float().unsqueeze(-1) * frequency
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ReferenceEncoder(nn.Module):
    def __init__(self, reference_dim=125, width=768, num_embodiments=1):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, reference_dim, width)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * width, width)
        self.W3 = CategorySpecificLinear(num_embodiments, width, width)
        self.pos_encoding = SinusoidalPositionalEncoding(width)

    def forward(self, reference, timestep, category_ids):
        if timestep.shape != (reference.shape[0],):
            raise ValueError("Expected one discrete flow timestep per sample")
        timestep = timestep[:, None].expand(-1, reference.shape[1])
        reference_emb = self.W1(reference, category_ids)
        time_emb = self.pos_encoding(timestep).to(reference_emb.dtype)
        result = self.W2(torch.cat((reference_emb, time_emb), dim=-1), category_ids)
        result = result * result.sigmoid()
        return self.W3(result, category_ids)
