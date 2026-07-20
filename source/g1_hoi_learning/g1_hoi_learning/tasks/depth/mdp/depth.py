"""Raycast depth observation for the depth-perception student (D435i head camera, simple_raycaster)."""

from __future__ import annotations

import math

import torch
import warp as wp

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from simple_raycaster.raycaster_v2 import MultiMeshRaycasterV2


class object_depth_b(ManagerTermBase):
    """Normalized head-D435i raycast depth image (num_envs, W*H).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        dev = env.device
        cam_pos = cfg.params["cam_pos"]
        cam_rpy = cfg.params["cam_rpy"]
        fov_h, fov_v = cfg.params["fov_deg"]
        width, height = cfg.params["resolution"]
        wp.init()
        self.rc = MultiMeshRaycasterV2(dev)
        self.rc.add_isaac_entity(env.scene["robot"])
        self.rc.add_isaac_entity(env.scene["object"])
        self.rc.add_isaac_static("/World/ground")
        # pinhole ray grid in camera frame: forward +x, image-right +y, image-up +z
        fh, fv = math.radians(fov_h), math.radians(fov_v)
        u = torch.linspace(math.tan(fh / 2), -math.tan(fh / 2), width, device=dev)
        v = torch.linspace(math.tan(fv / 2), -math.tan(fv / 2), height, device=dev)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        dirs = torch.stack([torch.ones_like(uu), uu, vv], dim=-1).reshape(-1, 3)
        self.dirs = (dirs / dirs.norm(dim=-1, keepdim=True)).contiguous()
        self.cam_pos = torch.tensor(cam_pos, device=dev)
        self.cam_quat = quat_from_euler_xyz(*(torch.tensor(a, device=dev) for a in cam_rpy)).reshape(4)
        self.torso = env.scene["robot"].find_bodies("torso_link")[0][0]

    def __call__(
        self,
        env: ManagerBasedEnv,
        cam_pos: tuple[float, float, float],
        cam_rpy: tuple[float, float, float],
        fov_deg: tuple[float, float],
        resolution: tuple[int, int],
        max_dist: float = 5.0,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        n, r = env.num_envs, self.dirs.shape[0]
        torso_pos = robot.data.body_pos_w[:, self.torso]
        torso_quat = robot.data.body_quat_w[:, self.torso]
        cam_p = torso_pos + quat_apply(torso_quat, self.cam_pos.expand(n, 3))
        cam_q = quat_mul(torso_quat, self.cam_quat.expand(n, 4))
        ray_dirs = quat_apply(cam_q[:, None, :].expand(n, r, 4), self.dirs[None].expand(n, r, 3))
        ray_starts = cam_p[:, None, :].expand(n, r, 3).contiguous()
        _, dist = self.rc.raycast_fused(ray_starts, ray_dirs.contiguous(), min_dist=0.02, max_dist=max_dist)
        return (dist / max_dist).clamp(0.0, 1.0)
