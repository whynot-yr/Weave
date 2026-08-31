"""Depth observation: head-camera depth degraded to D435i statistics, returned as a raw image."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg


def _blur(d, ksize=5, sigma=1.0):
    """Depthwise Gaussian blur (stereo smoothing; also yields edge flying-pixels). d: (N,H,W)."""
    ax = torch.arange(ksize, device=d.device, dtype=d.dtype) - (ksize - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    ker = torch.outer(g, g)
    ker = (ker / ker.sum()).view(1, 1, ksize, ksize)
    x = F.pad(d.unsqueeze(1), (ksize // 2,) * 4, mode="reflect")
    return F.conv2d(x, ker)[:, 0]


def _edge_holes(d, valid, edge_thresh=0.05, drop_p=0.5):
    """Holes at valid<->valid depth discontinuities ONLY (clip/hole borders excluded). d: (N,H,W)."""
    edge = torch.zeros_like(d, dtype=torch.bool)
    bx = ((d[:, :, 1:] - d[:, :, :-1]).abs() > edge_thresh) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, 1:] |= bx
    edge[:, :, :-1] |= bx
    by = ((d[:, 1:, :] - d[:, :-1, :]).abs() > edge_thresh) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, 1:, :] |= by
    edge[:, :-1, :] |= by
    edge &= torch.rand_like(d) < drop_p
    return F.max_pool2d(edge.float()[:, None], 3, 1, 1)[:, 0] > 0  # dilate -> blobby


def _blob_holes(d, max_dist, scale=12, base_p=0.01, range_p=0.06):
    """Spatially-correlated (blob) holes, more frequent far away. d: (N,H,W)."""
    N, H, W = d.shape
    low = torch.rand(N, 1, max(1, H // scale), max(1, W // scale), device=d.device)
    field = F.interpolate(low, (H, W), mode="bilinear", align_corners=False)[:, 0]
    return field < (base_p + range_p * (d / max_dist).clamp(0, 1))


def _degrade_depth(d, max_dist, noise_k=(0.005, 0.015), dropout=0.0,
                   min_z=0.25, blur_sigma=1.0, edge=(0.05, 0.5), blob=(12, 0.01, 0.06)):
    """Crisp raycast depth (m) -> D435i-like depth: Gaussian blur + axial noise + structured holes.

    Holes (sub-MinZ, >max clip / no-hit, real depth edges, blobs, salt-pepper) -> 0. Returns (N,H,W) m.
    """
    valid = (d > min_z) & (d < max_dist - 1e-3)
    hole = ~valid
    hole |= _edge_holes(d, valid, *edge)
    hole |= _blob_holes(d, max_dist, *blob)
    if dropout > 0.0:
        hole |= torch.rand_like(d) < dropout
    out = _blur(d, sigma=blur_sigma)
    kk = torch.empty(d.shape[0], 1, 1, device=d.device).uniform_(*noise_k)
    out = out + torch.randn_like(out) * (kk * out**2)   # axial noise, sigma ~ k*z^2
    return out.masked_fill(hole, 0.0).clamp(0.0, max_dist)


class object_depth_b(ManagerTermBase):
    """Head-camera depth -> D435i degradation -> normalized image (num_envs, H*W).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        p = cfg.params
        env.scene.sensors[p.get("sensor_name", "depth_cam")].set_robot(env.scene[p.get("robot_name", "robot")])
        self._dbg_state = None  # omni.ui live depth window

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_name: str = "depth_cam",
        max_dist: float = 3.0,
        min_z: float = 0.25,
        noise_k_range: tuple[float, float] = (0.005, 0.015),
        dropout_prob: float = 0.0,
        blur_sigma: float = 1.0,
        edge: tuple[float, float] = (0.05, 0.5),
        blob: tuple[float, float, float] = (12, 0.01, 0.06),
        robot_name: str = "robot",
        debug_vis: bool = False,
    ) -> torch.Tensor:
        depth = env.scene.sensors[sensor_name].data.output["distance_to_image_plane"]  # (N,H,W,1) m
        depth = depth.squeeze(-1).detach().clone()  # (N,H,W)
        depth = _degrade_depth(depth, max_dist, noise_k_range, dropout_prob, min_z, blur_sigma, edge, blob)
        if debug_vis:
            self._debug_show(depth[0], max_dist)
        return (depth / max_dist).flatten(1)  # (N, H*W), holes stay 0

    def _debug_show(self, d: torch.Tensor, max_dist: float) -> None:
        import matplotlib
        import omni.ui as ui

        img = (d.detach().float() / max_dist).clamp(0.0, 1.0).cpu().numpy()
        rgba = (matplotlib.colormaps["turbo"](img) * 255).astype("uint8")
        h, w = img.shape
        if self._dbg_state is None:
            provider = ui.ByteImageProvider()
            window = ui.Window("noised depth (env 0)", width=w * 3, height=h * 3 + 24)
            with window.frame:
                ui.ImageWithProvider(provider)
            self._dbg_state = (provider, window)
        self._dbg_state[0].set_bytes_data(rgba.flatten().tolist(), [w, h])
