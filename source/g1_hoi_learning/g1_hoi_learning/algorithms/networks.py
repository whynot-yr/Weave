from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


def l2normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Project onto the unit hypersphere along ``dim``."""
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(eps)


class Scaler(nn.Module):
    """Per-dimension learnable scale (SimBaV2): ``scaler * (init/scale) * x``.
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
        shift = torch.full_like(x[..., :1], self.c_shift)
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


class MLPEncoder(nn.Sequential):
    """Flat obs group -> latent_dim: ``Linear -> SiLU -> ... -> Linear(latent_dim)``.
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden_dims: list[int] = ()) -> None:
        layers: list[nn.Module] = []
        for a, b in zip([in_dim] + list(hidden_dims), hidden_dims):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers.append(nn.Linear(hidden_dims[-1] if hidden_dims else in_dim, latent_dim))
        super().__init__(*layers)


class ConvEncoder(nn.Module):
    """Flat (C*H*W) feature map -> conv stack -> latent_dim."""

    def __init__(self, in_dim: int, latent_dim: int, in_ch: int, hw: list[int],
                 channels: list[int] = (64, 64)) -> None:
        super().__init__()
        self.in_ch, self.h, self.w = in_ch, hw[0], hw[1]
        if in_ch * self.h * self.w != in_dim:
            raise ValueError(f"ConvEncoder: in_ch*H*W ({in_ch * self.h * self.w}) != group_dim ({in_dim}).")
        layers: list[nn.Module] = []
        c = in_ch
        for oc in channels:
            layers += [nn.Conv2d(c, oc, 3, stride=2, padding=1), nn.SiLU()]
            c = oc
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_ch, self.h, self.w)).flatten(1).shape[1]
        self.proj = nn.Sequential(nn.Linear(flat, latent_dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_ch, self.h, self.w)
        return self.proj(self.conv(x).flatten(1))


ENCODERS = {"mlp": MLPEncoder, "conv": ConvEncoder}


class GroupEncoder(nn.Module):
    """Per-observation-group encoder bank -> concatenated latents.
    """

    def __init__(self, group_dims: list[int], specs: list[dict], latent_dim: int) -> None:
        super().__init__()
        if len(group_dims) != len(specs):
            raise ValueError(f"group_dims ({len(group_dims)}) and specs ({len(specs)}) must align.")
        self.group_dims: list[int] = list(group_dims)
        self.in_features = sum(group_dims)
        self.out_features = latent_dim * len(group_dims)
        self.encoders = nn.ModuleList()
        for group_dim, spec in zip(group_dims, specs):
            spec = dict(spec)
            self.encoders.append(ENCODERS[spec.pop("type")](group_dim, latent_dim, **spec))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.group_dims, dim=-1)
        latents: list[torch.Tensor] = []
        for i, encoder in enumerate(self.encoders):
            latents.append(encoder(chunks[i]))
        return torch.cat(latents, dim=-1)


def build_group_backbone(
    obs: TensorDict,
    groups: list[str],
    encoder_hidden_dims: dict[str, Any],
    latent_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_blocks: int,
    expansion: int,
) -> nn.Sequential:
    """Per-group encoder bank -> SimBa backbone: ``nn.Sequential(GroupEncoder, SimBa)``."""
    group_dims = [obs[g].shape[-1] for g in groups]
    specs = [encoder_hidden_dims[g] for g in groups]
    encoder = GroupEncoder(group_dims, specs, latent_dim)
    backbone = SimBa(encoder.out_features, output_dim, hidden_dim, num_blocks, expansion)
    return nn.Sequential(encoder, backbone)
