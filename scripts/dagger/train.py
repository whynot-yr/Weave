"""Train depth Student-Weave online: student rollouts, current-window CNN/SimBa updates."""

from __future__ import annotations

import argparse
import random
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/dagger/train.yaml")
parser.add_argument("--teacher_checkpoint")
parser.add_argument("--teacher_agent_config")
parser.add_argument("--teacher_env_config")
initialization = parser.add_mutually_exclusive_group()
initialization.add_argument("--resume", help="v2 model checkpoint; resume optimizer, counters and RNG")
initialization.add_argument("--student_checkpoint", help="v1/v2 student weights; start a fresh training run")
parser.add_argument("--output_dir")
parser.add_argument("--num_envs", "--num-envs", type=int)
parser.add_argument("--micro_batch_size", type=int, help="Accumulation size; effective minibatch stays unchanged")
parser.add_argument("--seed", type=int)
parser.add_argument("--motion_files", nargs="+")
parser.add_argument(
    "--max_iterations", type=int, help="Limit iterations in this invocation without changing the schedule"
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args)
simulation_app = app.app

import g1_hoi_learning.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from common import environment_contract, make_env_cfg, read_yaml, teacher_config_paths
from g1_hoi_learning.algorithms.dagger.config import DaggerTrainConfig
from g1_hoi_learning.algorithms.dagger.networks import (
    TEACHER_GROUPS,
    StudentWeaveActor,
    TeacherWeaveActor,
    actor_spec,
)
from g1_hoi_learning.algorithms.dagger.runner import SCHEMA, DaggerRunner, file_sha256

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    config = read_yaml(args.config)
    resume_state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume_state:
        if resume_state.get("kind", "training") != "training":
            raise ValueError("--resume requires a full model_<iteration>.pt checkpoint, not student weights")
        if resume_state.get("schema") != SCHEMA:
            raise ValueError("Legacy training cannot resume this schedule; use --student_checkpoint to warm start")
        # The runner loads weights/optimizer later; retain only initialization metadata here.
        resume_state = {key: resume_state[key] for key in ("train_cfg", "teacher_info", "resolved_env")}
    # Resuming uses the saved schedule and teacher configs, not today's YAML defaults.
    settings = resume_state["train_cfg"] if resume_state else config.get("training", {})
    settings = dict(settings)
    for key in ("num_envs", "seed", "micro_batch_size"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    cfg = DaggerTrainConfig(**settings)
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    teacher_options = config.get("teacher", {})
    previous_teacher = resume_state["teacher_info"] if resume_state else {}
    checkpoint = args.teacher_checkpoint or previous_teacher.get("checkpoint") or teacher_options.get("checkpoint")
    if not checkpoint:
        raise ValueError("Supply --teacher_checkpoint or teacher.checkpoint in the training YAML")
    checkpoint = str(Path(checkpoint).resolve())
    if resume_state:
        agent_config, saved_env = previous_teacher["agent_cfg"], resume_state["resolved_env"]
    else:
        agent_path, env_path = teacher_config_paths(
            checkpoint,
            args.teacher_agent_config or teacher_options.get("agent_config"),
            args.teacher_env_config or teacher_options.get("env_config"),
        )
        raw_agent = read_yaml(agent_path)
        agent_config = raw_agent.get("agent", raw_agent)
        saved_env = read_yaml(env_path)
    overrides = {} if resume_state else config.get("env_overrides", {})
    if args.motion_files:
        overrides = dict(overrides)
        overrides["commands"] = {"motion": {"motion_files": args.motion_files}}
    device = args.device or "cuda:0"
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env_cfg = make_env_cfg(saved_env, cfg.num_envs, cfg.seed, device, overrides)
    run_dir = Path(
        args.output_dir
        or (
            str(Path(args.resume).resolve().parent)
            if args.resume
            else str(Path("logs/dagger") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
    ).resolve()
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"New run output must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(run_dir)
    env = gym.make("G1-Inspire-HOI-Student-v0", cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_config.get("clip_actions"))
        obs, _ = env.reset()
        for group in TEACHER_GROUPS:
            if obs[group].ndim != 2:
                raise ValueError(f"Expected flat teacher group: {group}")
        spec = actor_spec(agent_config, {g: obs[g].shape[-1] for g in TEACHER_GROUPS}, env.num_actions)
        teacher_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        teacher = TeacherWeaveActor(spec, teacher_state["model_state_dict"]).to(device)
        student = StudentWeaveActor(spec).to(device)
        student.initialize_from_teacher(teacher)
        teacher_info = {"checkpoint": checkpoint, "sha256": file_sha256(checkpoint), "agent_cfg": agent_config}
        runner = DaggerRunner(
            env,
            teacher,
            student,
            cfg,
            str(run_dir),
            teacher_info,
            environment_contract(env),
            env_cfg.to_dict(),
        )
        if args.resume:
            runner.load(args.resume)
        else:
            if args.student_checkpoint:
                runner.load_student(args.student_checkpoint)
            dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
            dump_yaml(
                str(run_dir / "params" / "dagger.yaml"),
                {
                    "training": asdict(cfg),
                    "teacher": teacher_info,
                    "student": spec,
                },
            )
        # Release checkpoint arrays before allocating the current rollout.
        del teacher_state, resume_state
        print(f"[DAgger] Teacher: {checkpoint}\n[DAgger] Output: {run_dir}", flush=True)
        samples = cfg.num_envs * cfg.rollout_steps
        print(
            f"[DAgger] Student execution; {samples} samples/rollout, "
            f"{cfg.num_learning_epochs} epochs x {cfg.num_mini_batches} minibatches = "
            f"{cfg.num_learning_epochs * cfg.num_mini_batches} optimizer updates/iteration, "
            f"microbatch={cfg.micro_batch_size}",
            flush=True,
        )
        runner.learn(obs, simulation_app.is_running, args.max_iterations)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Kit's native shutdown can terminate before Python prints an uncaught exception.
        traceback.print_exc()
        simulation_app.app.post_quit(1)
        raise
    finally:
        simulation_app.close()
