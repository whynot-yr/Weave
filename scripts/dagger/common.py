"""Configuration loading shared by DAgger scripts; import after AppLauncher."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path

import g1_hoi_learning
from g1_hoi_learning.algorithms.dagger.io import read_yaml  # noqa: F401
from g1_hoi_learning.algorithms.vision import DEPTH_FAR, DEPTH_NEAR, DEPTH_SIZE
from g1_hoi_learning.tasks.hoi.mdp.terminations import motion_clip_end
from g1_hoi_learning.tasks.hoi.student_env_cfg import G1StudentWeaveEnvCfg

from isaaclab.managers import TerminationTermCfg


def align_numeric_types(obj, data: dict):
    """IsaacLab's strict merger rejects float snapshots over integer-valued numeric defaults."""
    for key, value in data.items():
        if isinstance(obj, dict):
            if key not in obj:
                continue  # let the strict merger report unsupported schema keys
            current = obj[key]
        elif hasattr(obj, key):
            current = getattr(obj, key)
        else:
            continue
        if isinstance(value, Mapping):
            align_numeric_types(current, value)
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
        ):
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)


def relocate_package_assets(data: dict):
    """Relocate missing absolute package asset paths from teacher runs on other machines."""
    root = Path(g1_hoi_learning.__file__).resolve().parent
    for key, value in data.items():
        if isinstance(value, dict):
            relocate_package_assets(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    relocate_package_assets(item)
        elif key in {"asset_path", "usd_path"} and isinstance(value, str):
            marker = "/g1_hoi_learning/"
            if Path(value).is_absolute() and marker in value and not os.path.isfile(value):
                local = root / value.rsplit(marker, 1)[1]
                if local.is_file():
                    print(f"[DAgger] Relocated asset: {value} -> {local}", flush=True)
                    data[key] = str(local)


def teacher_config_paths(checkpoint: str, agent_path: str | None, env_path: str | None):
    params = Path(checkpoint).resolve().parent / "params"
    agent = Path(agent_path) if agent_path else params / "agent.yaml"
    env = Path(env_path) if env_path else params / "env.yaml"
    for path in (agent, env):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing teacher config {path}. Supply --teacher_agent_config and --teacher_env_config "
                "from the teacher run; network/observation settings must match its checkpoint."
            )
    return str(agent.resolve()), str(env.resolve())


def make_env_cfg(saved: dict, num_envs: int, seed: int, device: str, overrides: dict | None = None, evaluation=False):
    cfg = G1StudentWeaveEnvCfg()
    source = copy.deepcopy(saved.get("env", saved))
    source.get("scene", {}).pop("object", None)  # rebuilt from motion_files by the environment factory
    source.pop("log_dir", None)
    source.pop("seed", None)  # Base IsaacLab cfg seed is None; CLI/run seed is assigned below.
    relocate_package_assets(source)
    if "sim" in source and "actions" in source:  # full snapshot, rather than a partial Hydra override mapping
        for group in ("events", "rewards", "terminations"):
            saved_terms = source.get(group) or {}
            for name, term in vars(getattr(cfg, group)).items():
                if hasattr(term, "func") and name not in saved_terms:
                    setattr(getattr(cfg, group), name, None)
    align_numeric_types(cfg, source)
    cfg.from_dict(source)
    if overrides:
        # Observation/action/reference definitions remain the checkpoint's contract.
        allowed = {"commands", "events", "observations"}
        if set(overrides) - allowed:
            raise ValueError(f"env_overrides supports only {allowed}; use CLI for device/num_envs")
        for name, value in overrides.get("commands", {}).items():
            if name != "motion" or set(value) - {"motion_files", "rsi", "debug_vis"}:
                raise ValueError("Only motion files, RSI and debug visualization may be overridden")
        for name, value in overrides.get("observations", {}).items():
            if name != "robot_proprio" or set(value) - {"enable_corruption"}:
                raise ValueError("Only proprio enable_corruption may be overridden")
        cfg.from_dict(overrides)
    cfg.seed, cfg.sim.device, cfg.scene.num_envs = seed, device, num_envs
    cfg.scene.head_camera.height = cfg.scene.head_camera.width = DEPTH_SIZE
    cfg.scene.head_camera.data_types = ["depth"]
    cfg.scene.head_camera.update_period = 0.0
    cfg.scene.head_camera.spawn.clipping_range = (DEPTH_NEAR, DEPTH_FAR)
    cfg.num_rerenders_on_reset = 1
    cfg.sim.render_interval = cfg.decimation
    cfg.commands.motion.eval_mode = evaluation
    cfg.commands.motion.eval_clip_ids = []
    cfg.commands.motion.debug_vis = False
    if evaluation:
        cfg.commands.motion.rsi = False
        cfg.observations.robot_proprio.enable_corruption = False
        for name in vars(cfg.events):
            if not name.startswith("_"):
                setattr(cfg.events, name, None)
        cfg.terminations.clip_end = TerminationTermCfg(
            func=motion_clip_end, params={"command_name": "motion"}, time_out=True
        )
    elif not cfg.commands.motion.rsi:
        raise ValueError(
            "Training requires RSI to sample all clips; non-RSI training only samples each object's first clip"
        )
    # scene.object is intentionally MISSING until _make_env resolves motion-file assets.
    return cfg


def environment_contract(env) -> dict:
    raw = env.unwrapped
    cfg = raw.cfg.to_dict()
    action = raw.action_manager.get_term("joint_pos")
    return {
        "action_names": list(action._joint_names),
        "actions": cfg["actions"],
        "clip_actions": env.clip_actions,
        "robot": cfg["scene"]["robot"],
        "observations": cfg["observations"],
        "observation_terms": raw.observation_manager.active_terms,
        "motion": cfg["commands"]["motion"],
        "events": cfg["events"],
        "rewards": cfg["rewards"],
        "terminations": cfg["terminations"],
        "camera": cfg["scene"]["head_camera"],
        "dt": cfg["sim"]["dt"],
        "decimation": cfg["decimation"],
    }
