"""Evaluation-only fixed finger targets, without changing the policy interface."""
import torch


class FixedFingerControl:
    """Hold the 12 active fingers at zero radians; mimic targets follow normally."""

    def __init__(self, term, robot, clip_actions=None):
        expected = {f'{side}_{finger}_proximal_joint' for side in ('L', 'R')
                    for finger in ('index', 'middle', 'ring', 'pinky')}
        expected |= {f'{side}_thumb_proximal_{axis}_joint' for side in ('L', 'R')
                     for axis in ('yaw', 'pitch')}
        names = list(term._joint_names)
        if len(names) != 41 or not expected.issubset(names):
            raise ValueError('Fixed fingers require the 41-action Getup layout with 12 active fingers')
        self.indices = [i for i, name in enumerate(names) if name in expected]
        self.names = [names[i] for i in self.indices]
        raw = term.raw_actions
        scale = torch.broadcast_to(torch.as_tensor(term._scale, device=raw.device, dtype=raw.dtype), raw.shape)
        offset = torch.broadcast_to(torch.as_tensor(term._offset, device=raw.device, dtype=raw.dtype), raw.shape)
        scale, offset = scale[:, self.indices], offset[:, self.indices]
        if not torch.isfinite(scale).all() or (scale == 0).any() or not torch.isfinite(offset).all():
            raise ValueError('Finger action scale/offset must be finite and scale nonzero')
        self.actions = -offset / scale
        if not torch.isfinite(self.actions).all():
            raise ValueError('Fixed finger actions must be finite')
        if clip_actions is not None and (self.actions.abs() > clip_actions).any():
            raise ValueError('Wrapper action clipping would change the fixed finger targets')
        if term.cfg.clip is not None:
            bounds = term._clip[:, self.indices]
            if ((bounds[..., 0] > 0) | (bounds[..., 1] < 0)).any():
                raise ValueError('Action term clipping excludes the zero finger target')
        targets = dict.fromkeys(self.names, 0.0)
        for name, (driver, multiplier, offset) in term.cfg.mimic.items():
            if driver in targets:
                targets[name] = targets[driver] * multiplier + offset
        if len(targets) != 24:
            raise ValueError('Expected 12 active and 12 mimic finger targets')
        for name, target in targets.items():
            bounds = robot.data.joint_pos_limits[:, robot.joint_names.index(name)]
            if not torch.isfinite(torch.tensor(target)) or ((target < bounds[:, 0]) | (target > bounds[:, 1])).any():
                raise ValueError(f'Fixed finger target outside hard limits: {name}={target}')
        self.metadata = dict(finger_control='fixed', finger_action_indices=self.indices,
                             fixed_finger_joint_names=self.names,
                             fixed_finger_target_rad=[0.0] * len(self.names),
                             fixed_finger_targets_with_mimic_rad=targets)

    def apply(self, actions):
        result = actions.clone()
        result[:, self.indices] = self.actions
        return result
