"""Goal observation for the DAgger distillation.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, subtract_frame_transforms

from .commands import GoalMotionCommand


def goal_object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object position residual (current -> goal), in the robot anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    dpos_b = quat_apply_inverse(
        command.robot_anchor_quat_w, 
        command.goal_obj_pos_w - command.obj_pos_w
    )
    return dpos_b.view(env.num_envs, -1)


def goal_object_rot_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object orientation residual (current -> goal), in the robot anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    _, obj_b = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, 
        command.obj_pos_w, command.obj_quat_w
    )
    _, goal_b = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, 
        command.goal_obj_pos_w, command.goal_obj_quat_w
    )
    dquat = quat_mul(goal_b, quat_conjugate(obj_b))
    return matrix_from_quat(dquat)[..., :2].reshape(env.num_envs, -1)
