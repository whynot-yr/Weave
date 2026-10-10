"""WEAVE's 62 body + 63 object features, anchored to one actual pelvis pose.

Rot6d is the first two rotation matrix columns, flattened in row-major order.
All quaternions are wxyz. Positions are direct coordinates, never increments.
"""

import torch
from torch.nn import functional as F

REFERENCE_FIELDS = {
    "joint_pos": (0, 53),
    "pelvis_pos": (53, 56),
    "pelvis_rot6d": (56, 62),
    "object_pos": (62, 65),
    "object_rot6d": (65, 71),
    "contact": (71, 125),
}
PPO_OFFSETS = (0, 5, 10, 15, 20)


def matrix_from_quat(quat):
    if quat.shape[-1] != 4 or not torch.isfinite(quat).all():
        raise ValueError("Expected finite wxyz quaternion")
    if (quat.norm(dim=-1) < 1e-8).any():
        raise ValueError("Zero quaternion")
    w, x, y, z = F.normalize(quat, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def matrix_from_rot6d(value):
    """Gram-Schmidt projection with finite fallback for degenerate predictions."""
    columns = value.reshape(*value.shape[:-1], 3, 2)
    first, second = columns.unbind(-1)
    axis = torch.zeros_like(first)
    axis[..., 0] = 1
    first = torch.where((first.norm(dim=-1, keepdim=True) > 1e-6), first, axis)
    first = F.normalize(first, dim=-1)
    second = second - (second * first).sum(-1, keepdim=True) * first
    fallback = F.one_hot(first.abs().argmin(-1), 3).to(first.dtype)
    fallback = fallback - (fallback * first).sum(-1, keepdim=True) * first
    second = torch.where(second.norm(dim=-1, keepdim=True) > 1e-6, second, fallback)
    second = F.normalize(second, dim=-1)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def local_pose(position, quaternion, anchor_position, anchor_quaternion):
    inverse = matrix_from_quat(anchor_quaternion).transpose(-1, -2).unsqueeze(-3)
    pos = (inverse @ (position - anchor_position.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
    rot = inverse @ matrix_from_quat(quaternion)
    return pos, rot[..., :2].flatten(-2)


def encode_reference(reference, anchor_position, anchor_quaternion):
    pos, rot = local_pose(reference["pelvis_pos"], reference["pelvis_quat"], anchor_position, anchor_quaternion)
    obj_pos, obj_rot = local_pose(reference["object_pos"], reference["object_quat"], anchor_position, anchor_quaternion)
    value = torch.cat((reference["joint_pos"], pos, rot, obj_pos, obj_rot, reference["contact"]), dim=-1)
    if value.shape[-1] != 125 or not torch.isfinite(value).all():
        raise ValueError("Expected a finite reference125")
    return value


def split_reference(reference):
    if reference.shape[-1] != 125:
        raise ValueError("Expected reference125")
    return {"ref_robot": reference[..., :62], "ref_object": reference[..., 62:]}


def sanitize_reference(reference, contact_threshold=0.5):
    if reference.shape[-1] != 125 or not torch.isfinite(reference).all():
        raise ValueError("Invalid predicted reference")
    if not 0 < contact_threshold <= 1:
        raise ValueError("Contact threshold must lie in (0,1]")
    result = reference.clone()
    for segment in (slice(56, 62), slice(65, 71)):
        result[..., segment] = matrix_from_rot6d(result[..., segment])[..., :2].flatten(-2)
    contact = result[..., 71:]
    result[..., 71:] = torch.where(
        contact >= contact_threshold, 1.0, torch.where(contact <= -contact_threshold, -1.0, 0.0)
    )
    return result


def reanchor_reference(reference, source_pos, source_quat, current_pos, current_quat):
    inverse = matrix_from_quat(current_quat).transpose(-1, -2)
    relative = inverse @ matrix_from_quat(source_quat)
    translation = (inverse @ (source_pos - current_pos).unsqueeze(-1)).squeeze(-1)
    result = reference.clone()
    for pos, rot in ((slice(53, 56), slice(56, 62)), (slice(62, 65), slice(65, 71))):
        result[..., pos] = (relative.unsqueeze(-3) @ reference[..., pos, None]).squeeze(-1)
        result[..., pos] += translation.unsqueeze(-2)
        rotation = relative.unsqueeze(-3) @ matrix_from_rot6d(reference[..., rot])
        result[..., rot] = rotation[..., :2].flatten(-2)
    return result


def interpolate_reference(chunk, offsets, query):
    """Interpolate direct coordinates, project rotation blends, take nearest contacts.

    This coarse-chunk interpolation is an approximation; replan every control
    step to query exactly the original PPO offsets without interpolation.
    """
    times = torch.as_tensor(offsets, dtype=chunk.dtype, device=chunk.device)
    query = torch.as_tensor(query, dtype=chunk.dtype, device=chunk.device)
    if chunk.shape[-2:] != (len(times), 125) or (times[1:] <= times[:-1]).any():
        raise ValueError("Invalid reference time grid")
    if not torch.isfinite(query).all() or (query < times[0]).any() or (query > times[-1]).any():
        raise ValueError("Reference query outside prediction horizon")
    high = torch.searchsorted(times, query).clamp(1, len(times) - 1)
    low = high - 1
    weight = (query - times[low]) / (times[high] - times[low])
    result = torch.lerp(chunk[..., low, :], chunk[..., high, :], weight[:, None])
    nearest = torch.where(weight <= 0.5, low, high)
    result[..., 71:] = chunk[..., nearest, 71:]
    return sanitize_reference(result)


def ppo_reference_groups(reference):
    """Feature-major flattening, matching ObsGroup concatenation (310/315)."""
    if reference.shape[-2:] != (5, 125):
        raise ValueError("Expected five selected reference frames")
    body = torch.cat([reference[..., s].flatten(-2) for s in (slice(0, 53), slice(53, 56), slice(56, 62))], -1)
    obj = torch.cat([reference[..., s].flatten(-2) for s in (slice(62, 65), slice(65, 71), slice(71, 125))], -1)
    return {"ref_motion_body": body, "ref_motion_object": obj}


class ReferenceBuffer:
    def __init__(self, offsets, replan_steps=1):
        self.offsets = tuple(offsets)
        if replan_steps < 1 or replan_steps - 1 + max(PPO_OFFSETS) > max(offsets):
            raise ValueError("Replan interval exceeds prediction coverage")
        self.replan_steps = replan_steps
        self.reset()

    def reset(self):
        self.chunk = None
        self.start_step = None

    def needs_prediction(self, step):
        if step < 0 or (self.start_step is not None and step < self.start_step):
            raise ValueError("Control step moved backwards")
        return self.chunk is None or step - self.start_step >= self.replan_steps

    def install(self, chunk, position, quaternion, step):
        if chunk.shape != (len(self.offsets), 125):
            raise ValueError("Unexpected reference chunk shape")
        self.needs_prediction(step)
        matrix_from_quat(quaternion)
        if position.shape != (3,) or quaternion.shape != (4,) or not torch.isfinite(position).all():
            raise ValueError("Invalid pelvis anchor")
        self.chunk = sanitize_reference(chunk).detach().clone()
        self.anchor_pos, self.anchor_quat = position.detach().clone(), quaternion.detach().clone()
        self.start_step = step

    def groups(self, position, quaternion, step):
        if self.needs_prediction(step):
            raise RuntimeError("Prediction is missing or expired")
        selected = interpolate_reference(self.chunk, self.offsets, [step - self.start_step + x for x in PPO_OFFSETS])
        selected = reanchor_reference(selected, self.anchor_pos, self.anchor_quat, position, quaternion)
        return {key: value.unsqueeze(0) for key, value in ppo_reference_groups(selected).items()}
