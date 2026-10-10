"""Duck-typed WEAVE runtime validation and diagnostic reference replay."""

import torch

from .data.reference_codec import PPO_OFFSETS, encode_reference


def check_environment(env, config, source_info):
    metadata = source_info["metadata"]
    robot = env.scene["robot"]
    pairs = (
        ("joint_names", list(robot.joint_names)),
        ("body_names", list(robot.body_names)),
        ("action_names", list(env.action_manager.get_term("joint_pos")._joint_names)),
    )
    for key, actual in pairs:
        if metadata[key] != actual:
            raise ValueError(f"Robot/checkpoint {key} order mismatch")
    if abs(1 / env.step_dt - config.fps) > 1e-6:
        raise ValueError("Environment and reference policy FPS mismatch")
    command = env.command_manager.get_term("motion")
    if command.cfg.anchor_body_name != "pelvis" or tuple(command.cfg.future_offsets) != PPO_OFFSETS:
        raise ValueError("Unexpected PPO reference anchor or temporal offsets")


def replay_reference(command, offsets):
    """Only used by the explicitly selected motion diagnostic mode."""
    if command.time_steps.numel() != 1:
        raise ValueError("Diagnostic replay expects one environment")
    time = command.time_steps[:, None] + torch.tensor(offsets, device=command.time_steps.device)[None]
    time = time.minimum(command.motion.clip_lengths[command.env_clip][:, None] - 1)
    indices = time + command.motion.clip_starts[command.env_clip][:, None]
    buffer = command.motion.buffer
    origin = command._env.scene.env_origins[:, None]
    reference = {
        "joint_pos": buffer.joint_pos[indices],
        "pelvis_pos": buffer.body_pos_w[indices, command.anchor_index] + origin,
        "pelvis_quat": buffer.body_quat_w[indices, command.anchor_index],
        "object_pos": buffer.object_pos_w[indices] + origin,
        "object_quat": buffer.object_quat_w[indices],
        "contact": buffer.contact_label[indices].float(),
    }
    return encode_reference(reference, command.robot_anchor_pos_w, command.robot_anchor_quat_w)[0]
