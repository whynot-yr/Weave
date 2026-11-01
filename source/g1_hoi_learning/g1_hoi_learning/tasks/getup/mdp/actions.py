"""29 policy-controlled body joints and fixed Inspire finger targets."""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass


FINGER_DRIVERS = tuple(
    [f'{side}_{finger}_proximal_joint' for side in ('L', 'R')
     for finger in ('index', 'middle', 'ring', 'pinky')]
    + [f'{side}_thumb_proximal_{axis}_joint' for side in ('L', 'R') for axis in ('yaw', 'pitch')]
)
FINGER_MIMIC = {
    **{f'{side}_{finger}_intermediate_joint': (f'{side}_{finger}_proximal_joint', 1., 0.)
       for side in ('L', 'R') for finger in ('index', 'middle', 'ring', 'pinky')},
    **{f'{side}_thumb_{part}_joint': (f'{side}_thumb_proximal_pitch_joint', ratio, 0.)
       for side in ('L', 'R') for part, ratio in (('intermediate', 1.6), ('distal', 2.4))},
}


class FixedFingerJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.finger_driver_names = list(FINGER_DRIVERS)
        targets = dict.fromkeys(self.finger_driver_names, 0.)
        for name, (driver, multiplier, offset) in cfg.mimic.items():
            targets[name] = targets[driver] * multiplier + offset
        self.finger_joint_names = list(targets)
        self.finger_joint_ids, _ = self._asset.find_joints(self.finger_joint_names, preserve_order=True)
        if self.action_dim != 29 or len(self.finger_joint_ids) != 24:
            raise ValueError('Getup requires 29 body actions and 24 fixed finger joints')
        if set(self._joint_ids) & set(self.finger_joint_ids) or self._asset.num_joints != 53:
            raise ValueError('Body and finger joints must partition the 53-joint robot')
        self.fixed_finger_targets = torch.tensor(
            list(targets.values()), device=self.device, dtype=self.raw_actions.dtype
        ).repeat(self.num_envs, 1)
        limits = self._asset.data.joint_pos_limits[:, self.finger_joint_ids]
        if ((self.fixed_finger_targets < limits[..., 0]) |
                (self.fixed_finger_targets > limits[..., 1])).any():
            raise ValueError('Fixed finger targets exceed hard joint limits')

    def apply_actions(self):
        super().apply_actions()
        self._asset.set_joint_position_target(self.fixed_finger_targets, joint_ids=self.finger_joint_ids)


@configclass
class FixedFingerJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = FixedFingerJointPositionAction
    mimic: dict = FINGER_MIMIC
