"""Single-environment reference cache; no simulator/model dependencies."""

import torch

from .dp_codec import encode_reference, ppo_reference_groups, reanchor_reference
from .dp_protocol import OFFSETS
from .reference_window import window_indices


def validate_pose(position, quaternion):
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("Expected a single pelvis position[3] and quaternion[4]")
    if not torch.isfinite(position).all() or not torch.isfinite(quaternion).all():
        raise ValueError("Nonfinite actual pelvis pose")
    if not torch.isclose(quaternion.norm(), quaternion.new_tensor(1.0), atol=1e-3):
        raise ValueError("Actual pelvis quaternion must be unit length")


class DPReferenceBuffer:
    def __init__(self, replan_steps=20):
        if type(replan_steps) is not int or not 1 <= replan_steps <= 20:
            raise ValueError("Replanning interval must be in [1,20] for a 40-frame chunk and +20 lookahead")
        self.replan_steps = replan_steps
        self.reset()

    def reset(self):
        self.chunk = self.anchor_pos = self.anchor_quat = None
        self.start_step = None

    def needs_prediction(self, control_step):
        if type(control_step) is not int or control_step < 0:
            raise ValueError("Invalid control step")
        if self.start_step is not None and control_step < self.start_step:
            raise ValueError("Control step moved backwards; reset buffer on episode reset")
        return self.chunk is None or control_step - self.start_step >= self.replan_steps

    def install(self, chunk, position, quaternion, control_step):
        self.needs_prediction(control_step)
        validate_pose(position, quaternion)
        chunk = torch.as_tensor(chunk, dtype=torch.float32, device=position.device)
        if chunk.shape != (40, 125) or not torch.isfinite(chunk).all():
            raise ValueError("Expected finite reference [40,125]")
        self.chunk = chunk.detach().clone()
        self.anchor_pos, self.anchor_quat = position.detach().clone(), quaternion.detach().clone()
        self.start_step = control_step

    def reference_groups(self, position, quaternion, control_step):
        if self.needs_prediction(control_step):
            raise RuntimeError("Prediction missing or expired; replan before computing a PPO action")
        validate_pose(position, quaternion)
        indices = [control_step - self.start_step + offset for offset in OFFSETS]
        selected = self.chunk[indices]
        local = reanchor_reference(selected, self.anchor_pos, self.anchor_quat, position, quaternion)
        groups = ppo_reference_groups(local, offsets=(0, 1, 2, 3, 4))
        return {key: value.unsqueeze(0) for key, value in groups.items()}


def replay_reference_chunk(command):
    """Diagnostic mode ONLY: build a chunk from the original WEAVE motion buffer.

    The DP deployment path must not call this function. Coordinate conventions
    are world axes + env origin, matching actual body_pos_w used by the adapter.
    """
    if command.time_steps.numel() != 1:
        raise ValueError("Replay comparison requires exactly one environment")
    indices, _ = window_indices(
        command.time_steps,
        torch.arange(40, device=command.time_steps.device),
        command.motion.clip_lengths[command.env_clip],
        command.motion.clip_starts[command.env_clip],
    )
    buffer = command.motion.buffer
    origin = command._env.scene.env_origins[:, None, :]
    reference = {
        "joint_pos": buffer.joint_pos[indices],
        "pelvis_pos": buffer.body_pos_w[indices, command.anchor_index] + origin,
        "pelvis_quat": buffer.body_quat_w[indices, command.anchor_index],
        "object_pos": buffer.object_pos_w[indices] + origin,
        "object_quat": buffer.object_quat_w[indices],
        "contact": buffer.contact_label[indices].float(),
    }
    return encode_reference(reference, command.robot_anchor_pos_w, command.robot_anchor_quat_w)[0]
