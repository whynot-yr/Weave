"""Depth observation: head-camera depth degraded to D435i statistics -> frozen DeFM P4 features."""

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
    """Head-camera depth -> D435i degradation -> frozen DeFM (fp16) -> P4 features (num_envs, 128*Hf*Wf).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        p = cfg.params
        env.scene.sensors[p.get("sensor_name", "depth_cam")].set_robot(env.scene[p.get("robot_name", "robot")])
        from defm.model_factory import create_defm_model
        from defm.utils import preprocess_depth_batch

        self._defm = create_defm_model(p.get("defm_model", "defm_resnet18"), pretrained=True).eval().to(env.device)
        for q in self._defm.parameters():
            q.requires_grad_(False)
        self._preprocess = preprocess_depth_batch
        self._tgt = (int(p.get("defm_size", 224)) // 32) * 32

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
        defm_model: str = "defm_resnet18",
        defm_size: int = 512,
    ) -> torch.Tensor:
        depth = env.scene.sensors[sensor_name].data.output["distance_to_image_plane"]  # (N,H,W,1) m
        depth = depth.squeeze(-1).detach().clone()  # (N,H,W)
        depth = _degrade_depth(depth, max_dist, noise_k_range, dropout_prob, min_z, blur_sigma, edge, blob)
        x = self._preprocess(depth.unsqueeze(1), target_size=(self._tgt, self._tgt), device=depth.device)
        dev = "cuda" if depth.is_cuda else "cpu"
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16):
            p4 = self._defm(x)["dense_bifpn"]["P4"]  # (N, 128, Hf, Wf)
        return p4.float().flatten(1)  # (N, 128*Hf*Wf)
