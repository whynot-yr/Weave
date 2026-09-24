"""WEAVE DP protocol: RGB + state88 -> a current-pelvis-local reference125."""

import torch

from .reference_window import (
    local_pose_6d,
    matrix_from_6d,
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    rotation_6d,
)

PROTOCOL = "weave-dp-local-reference-v1"
STATE_DIM = 88
ACTION_DIM = 125
IMAGE_KEY = "observation.images.head"
REFERENCE_DIMS = {"joint_pos": 53, "pelvis_pos": 3, "pelvis_quat": 4, "object_pos": 3, "object_quat": 4, "contact": 54}


def active_joint_indices(joint_names, action_names):
    if len(joint_names) != 53 or len(set(joint_names)) != 53:
        raise ValueError("Expected 53 unique joint names")
    if len(action_names) != 41 or len(set(action_names)) != 41:
        raise ValueError("Expected 41 unique active joint names")
    return [joint_names.index(name) for name in action_names]


def encode_state(pelvis_quat, joint_pos, joint_vel, initial_pelvis_quat, active_indices):
    """HumanoidArena convention: remove initial heading, NOT current heading."""
    w, x, y, z = initial_pelvis_quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    heading = torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)
    ori = rotation_6d(quat_mul(quat_conjugate(heading), pelvis_quat))
    return torch.cat((ori, joint_pos[..., active_indices], joint_vel[..., active_indices]), -1)


def encode_reference(reference, actual_pos, actual_quat):
    """Convert ALL future poses using the one actual anchor at sample time t."""
    p, r = local_pose_6d(
        actual_pos[..., None, :], actual_quat[..., None, :], reference["pelvis_pos"], reference["pelvis_quat"]
    )
    op, ori = local_pose_6d(
        actual_pos[..., None, :], actual_quat[..., None, :], reference["object_pos"], reference["object_quat"]
    )
    return torch.cat((reference["joint_pos"], p, r, op, ori, reference["contact"]), -1)


def reanchor_reference(action, source_pos, source_quat, current_pos, current_quat):
    """Re-express a predicted chunk as pelvis moves during its execution.

    Positions are direct positions, never integrated increments. Rotations are
    projected to SO(3). Both anchors must use the same world/env-local axes.
    """
    src = matrix_from_quat(source_quat)
    dst_inverse = matrix_from_quat(current_quat).transpose(-1, -2)
    relative = dst_inverse @ src
    translation = (dst_inverse @ (source_pos - current_pos).unsqueeze(-1)).squeeze(-1)
    out = action.clone()
    for p, r in ((slice(53, 56), slice(56, 62)), (slice(62, 65), slice(65, 71))):
        out[..., p] = (relative[..., None, :, :] @ action[..., p, None]).squeeze(-1) + translation[..., None, :]
        rotation = relative[..., None, :, :] @ matrix_from_6d(action[..., r])
        out[..., r] = rotation[..., :2].flatten(-2)
    return out


def ppo_reference_groups(action, offsets=(0, 5, 10, 15, 20), contact_threshold=0.5):
    """Produce WEAVE feature-major PPO groups (310 body + 315 object for K=5).

    Call reanchor_reference first if the robot has moved since DP prediction.
    This only adapts reference groups, not object_state/robot_proprio.
    """
    if action.shape[-1] != ACTION_DIM or min(offsets) < 0 or max(offsets) >= action.shape[-2]:
        raise ValueError("Invalid reference shape or insufficient prediction horizon")
    if not 0 < contact_threshold <= 1:
        raise ValueError("Contact threshold must lie in (0,1]")
    a = action[..., list(offsets), :].clone()
    for s in (slice(56, 62), slice(65, 71)):
        a[..., s] = matrix_from_6d(a[..., s])[..., :2].flatten(-2)
    body = torch.cat([a[..., s].flatten(-2) for s in (slice(0, 53), slice(53, 56), slice(56, 62))], -1)
    contact = torch.where(
        a[..., 71:125] >= contact_threshold, 1.0, torch.where(a[..., 71:125] <= -contact_threshold, -1.0, 0.0)
    )
    obj = torch.cat((a[..., 62:65].flatten(-2), a[..., 65:71].flatten(-2), contact.flatten(-2)), -1)
    return {"ref_motion_body": body, "ref_motion_object": obj}
