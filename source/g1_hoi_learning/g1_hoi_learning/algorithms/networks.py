from __future__ import annotations

import torch
import torch.nn as nn


class SimBaBlock(nn.Module):
    """Pre-LN residual MLP block from SimBa (Lee et al., ICLR 2025).

    x -> LayerNorm -> Linear(d, d*m) -> ReLU -> Linear(d*m, d) -> (+ x)
    """

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class SimBa(nn.Module):
    """SimBa backbone: linear input projection -> N residual blocks -> post-LN -> linear head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[SimBaBlock(hidden_dim, expansion) for _ in range(num_blocks)])
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.output_proj.weight, gain=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.post_norm(x)
        return self.output_proj(x)

    def __getitem__(self, idx: int) -> nn.Module:
        # Compat shim for isaaclab_rl's policy exporter, which assumes
        # ``actor[0].in_features`` / ``actor[-1].out_features`` on an nn.Sequential.
        if idx == 0:
            return self.input_proj
        if idx == -1:
            return self.output_proj
        raise IndexError(f"SimBa only exposes indices 0 (input_proj) and -1 (output_proj); got {idx}")
