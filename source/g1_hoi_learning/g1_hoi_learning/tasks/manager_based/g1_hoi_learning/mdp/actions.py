from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class MimicJointPositionAction(JointPositionAction):
    """JointPositionAction that also writes URDF ``<mimic>`` joint targets in software.

    Every step, position targets are written for:
      - drive joints (matched by ``cfg.joint_names``): from the policy's processed action
      - mimic joints (keys of ``cfg.mimic``): ``driver_target * multiplier + offset``

    ``cfg.joint_names`` must exclude the mimic-target joints.
    """

    cfg: "MimicJointPositionActionCfg"

    def __init__(self, cfg: "MimicJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        mimic_names = list(cfg.mimic)
        driver_names = [d for d, _, _ in cfg.mimic.values()]

        self._mimic_joint_ids, _ = self._asset.find_joints(mimic_names, preserve_order=True)
        self._all_joint_ids = list(self._joint_ids) + self._mimic_joint_ids

        driver_action_idx = [self._joint_names.index(d) for d in driver_names]
        self._driver_action_idx = torch.tensor(driver_action_idx, device=env.device, dtype=torch.long)
        self._mimic_multipliers = torch.tensor(
            [m for _, m, _ in cfg.mimic.values()], device=env.device, dtype=torch.float32
        )
        self._mimic_offsets = torch.tensor(
            [o for _, _, o in cfg.mimic.values()], device=env.device, dtype=torch.float32
        )

    def apply_actions(self):
        drive_targets = self.processed_actions
        driver_targets = drive_targets[:, self._driver_action_idx]
        mimic_targets = driver_targets * self._mimic_multipliers + self._mimic_offsets
        all_targets = torch.cat([drive_targets, mimic_targets], dim=1)
        self._asset.set_joint_position_target(all_targets, joint_ids=self._all_joint_ids)


@configclass
class MimicJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`MimicJointPositionAction`."""

    class_type: type = MimicJointPositionAction

    mimic: dict = {}
    """Mapping ``mimic_joint_name -> (driver_joint_name, multiplier, offset)``.

    The mimic joint's position target is written every step as
    ``driver_target * multiplier + offset``. ``driver_joint_name`` must be in
    ``cfg.joint_names``; ``mimic_joint_name`` must NOT be.
    """
