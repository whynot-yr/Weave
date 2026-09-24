"""DP-play-only configuration. Never changes the original task defaults."""

import json
from pathlib import Path

import torch

from isaaclab.managers import TerminationTermCfg


def dp_fallen(env, min_height=0.25, max_gravity_z=-0.2):
    """Reference-independent coarse safety condition, not a task-success metric."""
    command = env.command_manager.get_term("motion")
    height = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height < min_height) | (env.scene["robot"].data.projected_gravity_b[:, 2] > max_gravity_z)


def dp_nonfinite_state(env):
    robot = env.scene["robot"].data
    return ~(
        torch.isfinite(robot.joint_pos).all(-1)
        & torch.isfinite(robot.joint_vel).all(-1)
        & torch.isfinite(robot.root_state_w).all(-1)
        & torch.isfinite(env.scene["object"].data.root_state_w).all(-1)
    )


def configure_dp_play(cfg, reference_file, clip_id, camera_config, episode_seconds, min_height=0.25):
    if clip_id < 0 or episode_seconds <= 0 or min_height <= 0:
        raise ValueError("Invalid clip id, episode length or minimum pelvis height")
    cfg.scene.num_envs = 1
    cfg.commands.motion.motion_files = [str(Path(reference_file).resolve())]
    cfg.commands.motion.eval_mode = True  # Clamp at end; NEVER reinitialize mid-episode.
    cfg.commands.motion.rsi = False
    cfg.commands.motion.eval_clip_ids = [clip_id]
    cfg.commands.motion.resampling_time_range = (1e9, 1e9)
    cfg.commands.motion.debug_vis = False
    cfg.num_rerenders_on_reset = 1
    cfg.episode_length_s = episode_seconds
    if list(cfg.commands.motion.future_offsets) != [0, 5, 10, 15, 20]:
        raise ValueError("Unexpected PPO reference offsets")
    # Use collection calibration, not a guessed camera with only matching size.
    recorded = json.loads(Path(camera_config).read_text())["env"]["scene"]["head_camera"]
    if (recorded["height"], recorded["width"]) != (224, 224):
        raise ValueError("Expected collection camera resolution 224x224")
    camera = cfg.scene.head_camera
    if recorded["prim_path"].split("/Robot/")[-1] != camera.prim_path.split("/Robot/")[-1]:
        raise ValueError("Collection camera is mounted on a different robot link/prim")
    camera.height, camera.width = recorded["height"], recorded["width"]
    camera.update_period = 0.0
    # RGB-only rendering preserves intrinsics/extrinsics; depth is unused by DP.
    camera.data_types = ["rgb"]
    for key in ("pos", "rot", "convention"):
        value = recorded["offset"][key]
        setattr(camera.offset, key, tuple(value) if isinstance(value, list) else value)
    for key in (
        "projection_type",
        "focal_length",
        "horizontal_aperture",
        "vertical_aperture",
        "horizontal_aperture_offset",
        "vertical_aperture_offset",
        "clipping_range",
        "focus_distance",
        "f_stop",
    ):
        value = recorded["spawn"][key]
        setattr(camera.spawn, key, tuple(value) if isinstance(value, list) else value)
    cfg.sim.render_interval = cfg.decimation
    # Deterministic evaluation: no startup/interval randomization or sensor noise.
    for name in list(vars(cfg.events)):
        if not name.startswith("_"):
            setattr(cfg.events, name, None)
    for group in vars(cfg.observations).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    reference_terms = {
        "bad_anchor_pos",
        "bad_anchor_ori",
        "bad_object_pos",
        "bad_object_ori",
        "bad_motion_body_pos_z_only",
        "bad_contact",
        "motion_clip_end",
    }
    disabled = []
    for name, term in list(vars(cfg.terminations).items()):
        if term is None or name.startswith("_"):
            continue
        function = getattr(term.func, "__name__", "")
        if function in reference_terms:
            setattr(cfg.terminations, name, None)
            disabled.append(name)
        elif function != "time_out":
            raise ValueError(f"Unreviewed termination {name}: {function}; inspect before DP deployment")
    cfg.terminations.dp_fallen = TerminationTermCfg(func=dp_fallen, params={"min_height": min_height})
    cfg.terminations.dp_nonfinite = TerminationTermCfg(func=dp_nonfinite_state)
    return {
        "disabled_reference_terminations": disabled,
        "camera": recorded,
        "safety_min_pelvis_height": min_height,
        "safety_max_gravity_z": -0.2,
    }
