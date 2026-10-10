# Adapted from NVIDIA Isaac-GR00T modules/dit.py (Apache-2.0).
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T block order using native Torch operations instead of diffusers imports.

Even blocks cross-attend to all DINO tokens. Odd blocks self-attend to reference
tokens. There is no state token, language branch, AdaLN-Zero gate or causal mask.
"""

from torch import nn
from torch.nn import functional as F

from .timestep_encoder import TimestepEncoder


class AdaLayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(width, 2 * width)
        self.norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)

    def forward(self, value, temb):
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        return self.norm(value) * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    """Matches diffusers Attention + AttnProcessor2_0 projections/dropout."""

    def __init__(self, width, heads, head_dim, context_dim, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.to_q = nn.Linear(width, heads * head_dim)
        self.to_k = nn.Linear(context_dim, heads * head_dim)
        self.to_v = nn.Linear(context_dim, heads * head_dim)
        self.to_out = nn.ModuleList((nn.Linear(heads * head_dim, width), nn.Dropout(dropout)))

    def forward(self, value, context=None):
        context = value if context is None else context
        batch = len(value)
        query = self.to_q(value).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        key = self.to_k(context).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        val = self.to_v(context).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        result = F.scaled_dot_product_attention(query, key, val, dropout_p=0.0, is_causal=False)
        result = result.transpose(1, 2).reshape(batch, -1, self.heads * self.head_dim)
        return self.to_out[1](self.to_out[0](result))


class GELU(nn.Module):
    def __init__(self, width, inner_dim):
        super().__init__()
        self.proj = nn.Linear(width, inner_dim)

    def forward(self, value):
        return F.gelu(self.proj(value), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.ModuleList(
            (
                GELU(width, width * 4),
                nn.Dropout(dropout),
                nn.Linear(width * 4, width),
                nn.Dropout(dropout),
            )
        )

    def forward(self, value):
        for layer in self.net:
            value = layer(value)
        return value


class BasicTransformerBlock(nn.Module):
    def __init__(self, width, heads, head_dim, context_dim, dropout):
        super().__init__()
        self.norm1 = AdaLayerNorm(width)
        self.attn1 = Attention(width, heads, head_dim, context_dim, dropout)
        self.final_dropout = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)
        self.ff = FeedForward(width, dropout)

    def forward(self, value, temb, context=None):
        result = self.attn1(self.norm1(value, temb), context)
        value = value + self.final_dropout(result)
        return value + self.ff(self.norm3(value))


class ReferenceDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timestep_encoder = TimestepEncoder(config.width)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    config.width,
                    config.num_heads,
                    config.head_dim,
                    config.image_dim if index % 2 == 0 else config.width,
                    config.dropout,
                )
                for index in range(config.num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(config.width, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(config.width, config.width * 2)
        self.proj_out_2 = nn.Linear(config.width, config.width)

    def forward(self, hidden_states, image_tokens, timestep):
        temb = self.timestep_encoder(timestep)
        for index, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states, temb, image_tokens if index % 2 == 0 else None)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        result = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(result)
