"""Evaluate pure student (or the frozen teacher baseline) on a single object's held-out clips."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/dagger/eval.yaml")
parser.add_argument("--checkpoint", required=True, help="Full DAgger checkpoint or lightweight stage weights")
parser.add_argument("--policy", choices=("student", "teacher"), default="student")
parser.add_argument("--teacher_checkpoint", help="Relocated teacher baseline checkpoint")
parser.add_argument("--motion_files", nargs="+", help="Held-out clips for one object")
parser.add_argument("--num_envs", type=int, help="Maximum camera environments per evaluation batch")
parser.add_argument("--seed", type=int)
parser.add_argument("--output_dir")
parser.add_argument("--no_video", action="store_true", help="Disable MP4 recording and evaluate in parallel batches")
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
from common import make_env_cfg, read_yaml
from g1_hoi_learning.algorithms.dagger.evaluation import ClipEvaluator, assign_eval_clips, summarize
from g1_hoi_learning.algorithms.dagger.networks import TEACHER_GROUPS, StudentWeaveActor, TeacherWeaveActor
from g1_hoi_learning.algorithms.dagger.runner import SCHEMA, file_sha256
from g1_hoi_learning.algorithms.dagger.video import ClipVideoRecorder, VideoConfig
from g1_hoi_learning.tasks.hoi.mdp.commands import MotionLoader

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    config = read_yaml(args.config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state.pop("replay", None)
    if state["schema"] != SCHEMA:
        raise ValueError("Expected a depth DAgger checkpoint")
    batch_size = args.num_envs if args.num_envs is not None else config.get("num_envs", 32)
    if batch_size < 1:
        raise ValueError("num_envs must be positive")
    video_settings = dict(config.get("video") or {})
    if args.no_video:
        video_settings["enabled"] = False
    video_cfg = VideoConfig(**video_settings)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    device = args.device or "cuda:0"
    motion_files = (
        args.motion_files or config.get("motion_files") or state["resolved_env"]["commands"]["motion"]["motion_files"]
    )
    if len(MotionLoader.object_names_of(motion_files)) != 1:
        raise ValueError(
            "Current Weave eval_mode supports one object per evaluation; pass --motion_files for one object"
        )
    names = []
    for filename in motion_files:
        with np.load(filename, allow_pickle=True) as data:
            names.extend(str(name) for name in data["motion_names"])
    if not names:
        raise ValueError("No evaluation clips")
    if args.policy == "student":
        policy = StudentWeaveActor(state["actor_spec"])
        policy.load_state_dict(state["student_state_dict"], strict=True)
    else:
        checkpoint = args.teacher_checkpoint or state["teacher_info"]["checkpoint"]
        if file_sha256(checkpoint) != state["teacher_info"]["sha256"]:
            raise ValueError("Teacher baseline differs from the training teacher")
        teacher_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        policy = TeacherWeaveActor(state["actor_spec"], teacher_state["model_state_dict"])
        del teacher_state
    policy.to(device).eval()
    saved_env = state["resolved_env"]
    clip_actions = state["environment_contract"]["clip_actions"]
    action_names = state["environment_contract"]["action_names"]
    spec = state["actor_spec"]
    del state  # release serialized replay; evaluation does not need it
    rows, times, metric_keys = [], [], []
    num_envs = 1 if video_cfg.enabled else min(batch_size, len(names))
    output = Path(args.output_dir or Path(args.checkpoint).resolve().parent / f"eval_{args.policy}")
    output.mkdir(parents=True, exist_ok=True)
    env_cfg = make_env_cfg(
        saved_env,
        num_envs,
        seed,
        device,
        {"commands": {"motion": {"motion_files": motion_files}}},
        evaluation=True,
    )
    env_cfg.commands.motion.eval_clip_ids = list(range(num_envs))
    if video_cfg.enabled:
        from video_recorder import EvaluationVideoCfg, read_rgb, warmup_rgb

        env_cfg.viewer.resolution = (video_cfg.width, video_cfg.height)
        env_cfg.viewer.env_index = 0
        # The recording camera tracks robot + object explicitly, not the UI's root-only camera.
        env_cfg.viewer.origin_type = "env"
        env_cfg.recorders = EvaluationVideoCfg()
        print(
            f"[Eval] Automatic video: {video_cfg.width}x{video_cfg.height}, "
            f"serial envs=1 (requested batch limit={batch_size})",
            flush=True,
        )
    env = gym.make("G1-Inspire-HOI-Student-v0", cfg=env_cfg, render_mode="rgb_array" if video_cfg.enabled else None)
    recorder = None
    report_metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "policy": args.policy,
        "motion_files": motion_files,
        "requested_num_clips": len(names),
        "max_batch_envs": batch_size,
        "actual_batch_envs": num_envs,
        "video": asdict(video_cfg),
    }

    def save_report():
        if not rows:
            return
        result = summarize(rows, metric_keys)
        result.update(report_metadata)
        result.update(
            all_requested_clips_evaluated=len(rows) == len(names),
            inference_batch_ms_mean=float(np.mean(times)) if times else None,
            inference_batch_ms_p95=float(np.percentile(times, 95)) if times else None,
        )
        temporary = output / "metrics.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(output / "metrics.json")

    try:
        env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
        if video_cfg.enabled:
            recorder = ClipVideoRecorder(output, video_cfg, env.unwrapped.step_dt, lambda: read_rgb(env.unwrapped))
            env.unwrapped.eval_video_term.recorder = recorder
            report_metadata["video"]["fps"] = recorder.fps
        command = env.unwrapped.command_manager.get_term("motion")
        if list(env.unwrapped.action_manager.get_term("joint_pos")._joint_names) != action_names:
            raise ValueError("Evaluation action joint order differs from training")
        for first in range(0, len(names), num_envs):
            ids = list(range(first, min(first + num_envs, len(names))))
            assign_eval_clips(command, ids)
            obs, _ = env.reset()
            if env.num_actions != spec["num_actions"] or any(
                obs[g].shape[-1] != spec["group_dims"][g] for g in TEACHER_GROUPS
            ):
                raise ValueError("Evaluation observation/action dimensions differ from training")
            evaluator = ClipEvaluator(env, names)
            try:
                if recorder is not None:
                    warmup_rgb(env.unwrapped)
                    recorder.start(ids[0], names[ids[0]])
                with torch.no_grad():
                    policy(obs)  # warm up kernels before measuring inference
                    for _ in range(evaluator.max_steps):
                        if not simulation_app.is_running():
                            break
                        if device.startswith("cuda"):
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        actions = policy(obs)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize(device)
                        times.append((time.perf_counter() - start) * 1000)
                        evaluator.collect(actions)
                        obs, _, dones, _ = env.step(actions)
                        evaluator.update(dones)
                        if evaluator.done:
                            break
            finally:
                batch_rows = evaluator.rows()[: len(ids)]
                if recorder is not None:
                    video_info = recorder.finish(complete=evaluator.done)
                    if video_info is not None:
                        batch_rows[0].update(video_info)
                rows.extend(batch_rows)
                metric_keys = list(evaluator.sums) + ["contact_lost_bodies_per_step", "action_change_mse"]
                save_report()
            print(f"[Eval] {len(rows)}/{len(names)} clips; successful={sum(r['success'] for r in rows)}", flush=True)
            if not simulation_app.is_running():
                break
    finally:
        try:
            if recorder is not None:
                recorder.finish(complete=False)
        finally:
            if hasattr(env.unwrapped, "eval_video_term"):
                env.unwrapped.eval_video_term.recorder = None
                del env.unwrapped.eval_video_term
            env.close()
    if not rows:
        raise RuntimeError("Evaluation did not collect any clips")
    result = summarize(rows, metric_keys)
    print(f"[Eval] Success={result['success_rate']:.4f}; metrics: {output / 'metrics.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.app.post_quit(1)
        raise
    finally:
        simulation_app.close()
