"""Raycast depth camera that poses the robot's articulation link meshes from ``body_link_pose_w``.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg
from isaaclab.utils import configclass

RaycastTargetCfg = MultiMeshRayCasterCfg.RaycastTargetCfg


@configclass
class ArticulationTargetCfg(RaycastTargetCfg):

    track_mesh_transforms: bool = False


class ArticulationRayCasterCamera(MultiMeshRayCasterCamera):

    cfg: ArticulationRayCasterCameraCfg

    def __init__(self, cfg: ArticulationRayCasterCameraCfg):
        super().__init__(cfg)
        self._robot: Articulation | None = None
        self._art_slots: list[tuple[int, int, list[str]]] = []  # (start, count, body_names) per art target
        self._art_idx: list[tuple[int, int, torch.Tensor]] | None = None

    def set_robot(self, robot: Articulation):
        self._robot = robot

    def _initialize_warp_meshes(self):
        super()._initialize_warp_meshes()
        # locate each articulation target's mesh-slot range and its per-slot body names (env_0 order)
        mesh_idx = 0
        self._art_slots = []
        for target_cfg in self._raycast_targets_cfg:
            count = self._num_meshes_per_env[target_cfg.prim_expr]
            if isinstance(target_cfg, ArticulationTargetCfg):
                prims = sim_utils.find_matching_prims(target_cfg.prim_expr)[:count]  # env_0 slot order
                self._art_slots.append((mesh_idx, count, [p.GetName() for p in prims]))
            mesh_idx += count

    def _resolve_art_idx(self):
        names = self._robot.body_names
        self._art_idx = []
        for start, count, body_names in self._art_slots:
            missing = [n for n in body_names if n not in names]
            if missing:
                raise KeyError(f"Articulation-target prims {missing} are not robot bodies. Robot bodies: {names}")
            idx = torch.tensor([names.index(n) for n in body_names], device=self.device, dtype=torch.long)
            self._art_idx.append((start, count, idx))

    def _update_buffers_impl(self, env_ids):
        # fill articulation-target mesh poses from body_link_pose_w before raycast
        if self._robot is not None and self._art_slots:
            if self._art_idx is None:
                self._resolve_art_idx()
            pose = self._robot.data.body_link_pose_w  # (num_envs, num_bodies, 7) -> pos(3), quat wxyz(4)
            for start, count, idx in self._art_idx:
                self._mesh_positions_w[:, start : start + count] = pose[:, idx, :3]
                self._mesh_orientations_w[:, start : start + count] = pose[:, idx, 3:7]
        super()._update_buffers_impl(env_ids)


@configclass
class ArticulationRayCasterCameraCfg(MultiMeshRayCasterCameraCfg):
    class_type: type = ArticulationRayCasterCamera
