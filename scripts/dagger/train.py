"""Train depth Student-Weave online: student rollouts, current-window CNN/SimBa updates."""

from __future__ import annotations

import argparse
import os
import random
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Containers can expose the NVIDIA Vulkan ICD in both search directories,
# causing duplicate GPU indices. Respect explicit loader overrides.
if not os.environ.get("VK_ICD_FILENAMES") and not os.environ.get("VK_DRIVER_FILES"):
    for icd_path in (Path("/etc/vulkan/icd.d/nvidia_icd.json"), Path("/usr/share/vulkan/icd.d/nvidia_icd.json")):
        if icd_path.is_file():
            os.environ["VK_ICD_FILENAMES"] = str(icd_path)
            print(f"[DAgger] Vulkan ICD: {icd_path}", flush=True)
            break

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
parser.add_argument(
    "--distributed", action="store_true", help="Enable DDP under torchrun; auto-enabled for WORLD_SIZE > 1"
)
parser.add_argument("--num_envs", "--num-envs", type=int, help="Environment count PER GPU/rank")
parser.add_argument("--micro_batch_size", type=int, help="Accumulation size; effective minibatch stays unchanged")
parser.add_argument("--seed", type=int)
parser.add_argument("--motion_files", nargs="+")
parser.add_argument(
    "--max_iterations", type=int, help="Limit iterations in this invocation without changing the schedule"
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
world_size = int(os.environ.get("WORLD_SIZE", "1"))
args.distributed = args.distributed or world_size > 1
if args.distributed:
    if world_size < 2:
        parser.error("--distributed requires torchrun with --nproc-per-node >= 2")
    if args.device is not None and args.device.startswith("cpu"):
        parser.error("Distributed IsaacLab simulation requires CUDA")
    # AppLauncher binds both physics and camera rendering before SimulationApp starts.
    args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
args.enable_cameras = True
app = AppLauncher(args)
simulation_app = app.app

import g1_hoi_learning.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from common import environment_contract, make_env_cfg, read_yaml, teacher_config_paths
from g1_hoi_learning.algorithms.dagger.config import DaggerTrainConfig
from g1_hoi_learning.algorithms.dagger.distributed import DistributedContext
from g1_hoi_learning.algorithms.dagger.networks import (
    TEACHER_GROUPS,
    StudentWeaveActor,
    TeacherWeaveActor,
    actor_spec,
)
from g1_hoi_learning.algorithms.dagger.runner import SCHEMA, DaggerRunner, file_sha256

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main(context):
    context.require_same(
        (args.config, args.resume, args.student_checkpoint, args.output_dir, args.motion_files, args.max_iterations),
        "input paths and iteration limit",
    )
    config = context.all_call(lambda: read_yaml(args.config))
    resume_state = (
        context.all_call(lambda: torch.load(args.resume, map_location="cpu", weights_only=False))
        if args.resume
        else None
    )
    if resume_state:
        if resume_state.get("kind", "training") != "training":
            raise ValueError("--resume requires a full model_<iteration>.pt checkpoint, not student weights")
        if resume_state.get("schema") != SCHEMA:
            raise ValueError("Legacy training cannot resume this schedule; use --student_checkpoint to warm start")
        if resume_state.get("world_size", 1) != context.world_size:
            raise ValueError("Resume requires the same world size; use --student_checkpoint for a new run")
        # The runner loads weights/optimizer later; retain only initialization metadata here.
        resume_state = {key: resume_state[key] for key in ("train_cfg", "teacher_info", "resolved_env")}
    # Resuming uses the saved schedule and teacher configs, not today's YAML defaults.
    settings = resume_state["train_cfg"] if resume_state else config.get("training", {})
    settings = dict(settings)
    if resume_state:
        # Save controls come from today's YAML; learning configuration stays checkpoint-owned.
        for key in ("save_interval", "keep_last_checkpoints"):
            if key in config.get("training", {}):
                settings[key] = config["training"][key]
    for key in ("num_envs", "seed", "micro_batch_size"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    cfg = context.all_call(lambda: DaggerTrainConfig(**settings))
    context.require_same(asdict(cfg), "per-rank training configuration")
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
        agent_path, env_path = context.all_call(
            lambda: teacher_config_paths(
                checkpoint,
                args.teacher_agent_config or teacher_options.get("agent_config"),
                args.teacher_env_config or teacher_options.get("env_config"),
            )
        )
        raw_agent = context.all_call(lambda: read_yaml(agent_path))
        agent_config = raw_agent.get("agent", raw_agent)
        saved_env = context.all_call(lambda: read_yaml(env_path))
    overrides = {} if resume_state else config.get("env_overrides", {})
    if args.motion_files:
        overrides = dict(overrides)
        overrides["commands"] = {"motion": {"motion_files": args.motion_files}}
    device = str(context.device)
    rank_seed = cfg.seed + context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    env_cfg = context.all_call(lambda: make_env_cfg(saved_env, cfg.num_envs, rank_seed, device, overrides))

    def prepare_output():
        root = Path(
            args.output_dir
            or (
                str(Path(args.resume).resolve().parent)
                if args.resume
                else str(Path("logs/dagger") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            )
        ).resolve()
        if not args.resume and root.exists() and any(root.iterdir()):
            raise FileExistsError(f"New run output must be empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    run_dir = Path(context.rank_zero_call(prepare_output))
    env_cfg.log_dir = str(run_dir / f"rank_{context.rank}" / "simulation") if context.distributed else str(run_dir)
    context.all_call(lambda: Path(env_cfg.log_dir).mkdir(parents=True, exist_ok=True))
    env = context.all_call(lambda: gym.make("G1-Inspire-HOI-Student-v0", cfg=env_cfg))
    try:
        env = context.all_call(lambda: RslRlVecEnvWrapper(env, clip_actions=agent_config.get("clip_actions")))
        obs, _ = context.all_call(env.reset)

        def validate_actor_spec():
            for group in TEACHER_GROUPS:
                if obs[group].ndim != 2:
                    raise ValueError(f"Expected flat teacher group: {group}")
            return actor_spec(agent_config, {g: obs[g].shape[-1] for g in TEACHER_GROUPS}, env.num_actions)

        spec = context.all_call(validate_actor_spec)
        teacher_state = context.all_call(lambda: torch.load(checkpoint, map_location="cpu", weights_only=False))
        teacher = context.all_call(
            lambda state=teacher_state: TeacherWeaveActor(spec, state["model_state_dict"]).to(device)
        )
        # Identical initialization, then independent environment/minibatch random streams.
        torch.manual_seed(cfg.seed)
        student = context.all_call(lambda: StudentWeaveActor(spec).to(device))
        context.all_call(lambda: student.initialize_from_teacher(teacher))
        teacher_info = {
            "checkpoint": checkpoint,
            "sha256": context.all_call(lambda: file_sha256(checkpoint)),
            "agent_cfg": agent_config,
        }
        runner = DaggerRunner(
            env,
            teacher,
            student,
            cfg,
            str(run_dir),
            teacher_info,
            context.all_call(lambda: environment_contract(env)),
            context.all_call(env_cfg.to_dict),
            context,
        )
        if args.resume:
            runner.load(args.resume)
        else:
            if args.student_checkpoint:
                runner.load_student(args.student_checkpoint)

            def write_parameters():
                dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
                dump_yaml(
                    str(run_dir / "params" / "dagger.yaml"),
                    {
                        "training": asdict(cfg),
                        "teacher": teacher_info,
                        "student": spec,
                        "distributed": {"world_size": context.world_size, "num_envs_per_rank": cfg.num_envs},
                    },
                )

            context.rank_zero_call(write_parameters)
            random.seed(rank_seed)
            np.random.seed(rank_seed)
            torch.manual_seed(rank_seed)
        # Release checkpoint arrays before allocating each rank's current rollout.
        del teacher_state, resume_state
        print(f"[DAgger rank {context.rank}] device={device} envs={cfg.num_envs} seed={rank_seed}", flush=True)
        if context.rank == 0:
            samples = cfg.num_envs * cfg.rollout_steps
            print(
                f"[DAgger] Teacher: {checkpoint}\n[DAgger] Output: {run_dir}\n"
                f"[DAgger] world_size={context.world_size}, envs_per_rank={cfg.num_envs}, "
                f"global_envs={cfg.num_envs * context.world_size}; "
                f"{samples} samples/rank/rollout, global_samples={samples * context.world_size}; "
                f"{cfg.num_learning_epochs} epochs x {cfg.num_mini_batches} minibatches = "
                f"{cfg.num_learning_epochs * cfg.num_mini_batches} synchronized updates/iteration; "
                f"microbatch_per_rank={cfg.micro_batch_size}; checkpoints every {cfg.save_interval} iterations, "
                f"retention={cfg.keep_last_checkpoints if cfg.keep_last_checkpoints is not None else 'all'}",
                flush=True,
            )
        runner.learn(obs, simulation_app.is_running, args.max_iterations)
    finally:
        env.close()


if __name__ == "__main__":
    context = None
    try:
        context = DistributedContext.initialize(args.device or "cuda:0")
        main(context)
    except Exception:
        # Kit's native shutdown can terminate before Python prints an uncaught exception.
        traceback.print_exc()
        simulation_app.app.post_quit(1)
        raise
    finally:
        if context is not None:
            context.close()
        simulation_app.close()
