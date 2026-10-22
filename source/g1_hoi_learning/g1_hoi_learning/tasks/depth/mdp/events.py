"""Depth-camera domain randomization events (extrinsics + intrinsics).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase


class randomize_camera_extrinsics(ManagerTermBase):
    """Per-reset Gaussian perturbation of the depth camera mount pose."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_name"]]
        self._nom_pos = self._sensor._offset_pos.clone()
        self._nom_quat = self._sensor._offset_quat.clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        sensor_name: str,
        pos_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        dev, n = self._sensor.device, len(env_ids)
        if any(s > 0.0 for s in pos_noise_std):
            std = torch.tensor(pos_noise_std, device=dev)
            self._sensor._offset_pos[env_ids] = self._nom_pos[env_ids] + torch.randn(n, 3, device=dev) * std
        if any(s > 0.0 for s in rot_noise_std):
            std = torch.tensor(rot_noise_std, device=dev)
            rpy = torch.randn(n, 3, device=dev) * std
            noise_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            self._sensor._offset_quat[env_ids] = math_utils.quat_mul(self._nom_quat[env_ids], noise_quat)


class randomize_camera_intrinsics(ManagerTermBase):
    """Per-reset Gaussian perturbation of the depth camera focal length / aperture."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_name"]]
        self._nom_K = self._sensor._data.intrinsic_matrices.clone()  # nominal intrinsics
        p = self._sensor.cfg.pattern_cfg
        self._focal, self._hap, self._vap = p.focal_length, p.horizontal_aperture, p.vertical_aperture

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        sensor_name: str,
        focal_length_noise_std: float = 0.0,
        aperture_noise_std: float = 0.0,
    ):
        if focal_length_noise_std <= 0.0 and aperture_noise_std <= 0.0:
            return
        dev, n = self._sensor.device, len(env_ids)
        K = self._nom_K[env_ids].clone()  # (n, 3, 3)
        if focal_length_noise_std > 0.0:  # focal up -> fx,fy up -> narrower FOV
            scale = (self._focal + torch.randn(n, device=dev) * focal_length_noise_std) / self._focal
            K[:, 0, 0] *= scale
            K[:, 1, 1] *= scale
        if aperture_noise_std > 0.0:  # aperture up -> fx,fy down -> wider FOV
            d_hap = torch.randn(n, device=dev) * aperture_noise_std
            d_vap = d_hap * (self._vap / self._hap)
            K[:, 0, 0] *= self._hap / (self._hap + d_hap)
            K[:, 1, 1] *= self._vap / (self._vap + d_vap)
        self._sensor.set_intrinsic_matrices(K, focal_length=self._focal, env_ids=env_ids)
