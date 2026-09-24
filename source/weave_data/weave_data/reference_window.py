"""Offline WEAVE future-window operations, independent of Isaac Sim.

Sampling follows MotionCommand._refresh_caches; the pose operation is
the subtract_frame_transforms + matrix_from_quat(...)[..., :2] operation used by
WEAVE's future-reference observations. The original PPO code remains untouched.
Quaternions are wxyz; 6D rotations flatten
the first two matrix columns in ROW-major order (not column concatenation).
"""

import torch
import torch.nn.functional as F


def window_indices(time, offsets, lengths, starts=None):
    """Per-clip future indices and padding mask. Leading dimensions broadcast."""
    raw = time[..., None] + offsets
    pad = (raw < 0) | (raw >= lengths[..., None])
    indices = raw.clamp_min(0).minimum(lengths[..., None] - 1)
    if starts is not None:
        indices = indices + starts[..., None]
    return indices, pad


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    av, bv = torch.broadcast_tensors(av, bv)
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def quat_apply(q, v):
    xyz, v = torch.broadcast_tensors(q[..., 1:], v)
    t = 2 * torch.linalg.cross(xyz, v)
    return v + q[..., :1] * t + torch.linalg.cross(xyz, t)


def matrix_from_quat(q):
    w, x, y, z = q.unbind(-1)
    s = 2 / (q * q).sum(-1)
    return torch.stack(
        (
            1 - s * (y * y + z * z),
            s * (x * y - z * w),
            s * (x * z + y * w),
            s * (x * y + z * w),
            1 - s * (x * x + z * z),
            s * (y * z - x * w),
            s * (x * z - y * w),
            s * (y * z + x * w),
            1 - s * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation_6d(q):
    return matrix_from_quat(q)[..., :2].reshape(*q.shape[:-1], 6)


def matrix_from_6d(r):
    """Project predicted 6D columns onto SO(3), including degenerate predictions."""
    columns = r.reshape(*r.shape[:-1], 3, 2)
    a, b = columns.unbind(-1)
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    a = F.normalize(torch.where(a.norm(dim=-1, keepdim=True) > 1e-6, a, fallback), dim=-1)
    b = b - (a * b).sum(-1, keepdim=True) * a
    axis = F.one_hot(a.abs().argmin(-1), 3).to(a)
    fallback = axis - (a * axis).sum(-1, keepdim=True) * a
    b = F.normalize(torch.where(b.norm(dim=-1, keepdim=True) > 1e-6, b, fallback), dim=-1)
    return torch.stack((a, b, torch.linalg.cross(a, b)), -1)


def local_pose(anchor_pos, anchor_quat, target_pos, target_quat):
    """Same operation as WEAVE's subtract_frame_transforms (unit quaternions)."""
    inverse = quat_conjugate(anchor_quat) / anchor_quat.square().sum(-1, keepdim=True).clamp_min(1e-9)
    return quat_apply(inverse, target_pos - anchor_pos), quat_mul(inverse, target_quat)


def local_pose_6d(anchor_pos, anchor_quat, target_pos, target_quat):
    pos, quat = local_pose(anchor_pos, anchor_quat, target_pos, target_quat)
    return pos, rotation_6d(quat)
