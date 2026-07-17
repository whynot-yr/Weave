from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Project onto the unit hypersphere along ``dim``."""
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class Scaler(nn.Module):
    """Per-dimension learnable scale (SimBaV2): ``scaler * (init/scale) * x``.

    ``scaler`` is initialized to ``scale`` so the effective initial multiplier is ``init``;
    the ``init/scale`` split decouples the parameter magnitude from the effective scale.
    """

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scale = float(init) / float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler * self.forward_scale * x


class HyperDense(nn.Module):
    """Bias-free linear whose weight is L2-normalized per output unit (SimBaV2)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1))


class HyperEmbedder(nn.Module):
    """Project the input onto the unit hypersphere: shift -> l2norm -> dense -> scaler -> l2norm."""

    def __init__(self, in_dim: int, hidden_dim: int, scaler_init: float, scaler_scale: float, c_shift: float) -> None:
        super().__init__()
        self.c_shift = float(c_shift)
        self.dense = HyperDense(in_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = x.new_full((*x.shape[:-1], 1), self.c_shift)
        x = torch.cat([x, shift], dim=-1)
        x = l2normalize(x)
        x = self.scaler(self.dense(x))
        return l2normalize(x)


class HyperMLP(nn.Module):
    """HyperDense(up) -> Scaler -> ReLU -> HyperDense(down) -> l2norm (SimBaV2)."""

    def __init__(self, hidden_dim: int, out_dim: int, expansion: int, scaler_init: float, scaler_scale: float) -> None:
        super().__init__()
        mid = hidden_dim * expansion
        self.up = HyperDense(hidden_dim, mid)
        self.scaler = Scaler(mid, scaler_init, scaler_scale)
        self.down = HyperDense(mid, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.scaler(self.up(x)))
        return l2normalize(self.down(x))


class HyperLERPBlock(nn.Module):
    """SimBaV2 residual block with learnable spherical interpolation::

        x <- l2normalize( x + alpha * (HyperMLP(x) - x) )
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
    ) -> None:
        super().__init__()
        self.mlp = HyperMLP(
            hidden_dim,
            hidden_dim,
            expansion,
            scaler_init / math.sqrt(expansion),
            scaler_scale / math.sqrt(expansion),
        )
        self.alpha = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.mlp(x)
        return l2normalize(residual + self.alpha(h - residual))


class SimBa(nn.Module):
    """SimBaV2 backbone with hyperspherical normalization (arXiv:2502.15280).

    HyperEmbedder -> N x HyperLERPBlock -> (HyperDense -> Scaler -> Linear) head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion: int = 4,
        c_shift: float = 3.0,
    ) -> None:
        super().__init__()
        self.in_features = input_dim
        self.out_features = output_dim

        scaler_init = math.sqrt(2.0 / hidden_dim)
        scaler_scale = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (num_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)

        self.embedder = HyperEmbedder(input_dim, hidden_dim, scaler_init, scaler_scale, c_shift)
        self.blocks = nn.Sequential(
            *[
                HyperLERPBlock(hidden_dim, expansion, scaler_init, scaler_scale, alpha_init, alpha_scale)
                for _ in range(num_blocks)
            ]
        )
        # output head: HyperDense -> Scaler -> Linear (leaves the sphere)
        self.head_dense = HyperDense(hidden_dim, hidden_dim)
        self.head_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.output_proj.weight, gain=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        x = self.blocks(x)
        x = self.head_scaler(self.head_dense(x))
        return self.output_proj(x)
