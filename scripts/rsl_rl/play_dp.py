# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone synchronous DP -> WEAVE PPO rollout. Original play.py is untouched."""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-Inspire-HOI-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--reference_file", type=Path, required=True, help="Single-object NPZ used only for initialization")
parser.add_argument("--clip-id", type=int, default=0)
parser.add_argument("--camera-config", type=Path, required=True, help="Collection's resolved_config.json")
parser.add_argument("--ppo-config", type=Path, help="Saved agent.yaml; defaults to checkpoint's params/agent.yaml")
parser.add_argument("--dp-url", default="http://127.0.0.1:8765")
parser.add_argument("--dp-timeout", type=float, default=30.0)
parser.add_argument("--replan-steps", type=int, default=20)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--episode-seconds", type=float, default=20.0)
parser.add_argument("--max-steps", type=int, default=1000, help="Total physics control steps across episodes")
parser.add_argument("--min-pelvis-height", type=float, default=0.25)
parser.add_argument(
    "--reference-source",
    choices=["dp", "motion"],
    default="dp",
    help="motion is diagnostic reference replay, NEVER an automatic fallback",
)
parser.add_argument("--output", type=Path, required=True, help="New directory for configuration and JSONL trace")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs != 1 or min(args_cli.episodes, args_cli.max_steps) <= 0:
    parser.error("V1 requires one environment, positive episodes and max-steps")
if not args_cli.checkpoint or not Path(args_cli.checkpoint).is_file():
    parser.error("Specify an existing trusted local PPO --checkpoint")
if not args_cli.reference_file.is_file() or not args_cli.camera_config.is_file():
    parser.error("Reference NPZ and collection camera config must exist")
if args_cli.output.exists():
    parser.error("Output directory already exists; refusing to overwrite")
if not 1 <= args_cli.replan_steps <= 20:
    parser.error("--replan-steps must be in [1,20]")
if any(
    not math.isfinite(x) or x <= 0 for x in (args_cli.episode_seconds, args_cli.dp_timeout, args_cli.min_pelvis_height)
):
    parser.error("Episode duration, DP timeout and minimum pelvis height must be finite and positive")
if args_cli.clip_id < 0 or not 0 <= args_cli.seed < 2**32 - args_cli.episodes:
    parser.error("Invalid clip-id or seed range")
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# IsaacLab/task imports must follow AppLauncher.
import json
import time
import traceback

import g1_hoi_learning.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import yaml
from g1_hoi_learning.rollout.dp_play_config import configure_dp_play
from rsl_rl.runners import OnPolicyRunner
from weave_data.dp_codec import encode_state
from weave_data.dp_http_client import DPHttpClient
from weave_data.dp_reference_buffer import DPReferenceBuffer, replay_reference_chunk

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def check_environment(raw, protocol):
    robot = raw.scene["robot"]
    action_names = list(raw.action_manager.get_term("joint_pos")._joint_names)
    for label, actual in (
        ("joint_names", list(robot.joint_names)),
        ("body_names", list(robot.body_names)),
        ("action_names", action_names),
    ):
        if actual != protocol[label]:
            raise ValueError(f"Environment/checkpoint {label} order mismatch")
    if abs(1 / raw.step_dt - protocol["fps"]) > 1e-6:
        raise ValueError("Environment control rate differs from DP training data")
    if raw.command_manager.get_term("motion").cfg.anchor_body_name != "pelvis":
        raise ValueError("DP protocol requires pelvis anchor")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    ppo_config = args_cli.ppo_config or Path(args_cli.checkpoint).resolve().parent / "params/agent.yaml"
    if not ppo_config.is_file():
        raise FileNotFoundError(f"PPO training configuration missing: {ppo_config}; specify --ppo-config")
    saved_agent = yaml.safe_load(ppo_config.read_text())
    if saved_agent.get("class_name") != "OnPolicyRunner":
        raise ValueError("Only the original WEAVE OnPolicyRunner checkpoint is supported")
    agent_cfg.from_dict(saved_agent)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.policy.compile = False  # Avoid compilation latency; same PPO weights/math.
    with np.load(args_cli.reference_file, allow_pickle=True) as reference:
        objects = [str(x) for x in reference["object_names"]]
        if len(set(objects)) != 1 or not 0 <= args_cli.clip_id < len(reference["motion_lengths"]):
            raise ValueError("V1 requires single-object reference NPZ and a valid clip-id")
        reference_fps = float(np.asarray(reference["fps"]).reshape(-1)[0])
    if abs(reference_fps - 1 / (env_cfg.sim.dt * env_cfg.decimation)) > 1e-6:
        raise ValueError("Reference initialization motion fps does not match environment")
    adjustments = configure_dp_play(
        env_cfg,
        args_cli.reference_file,
        args_cli.clip_id,
        args_cli.camera_config,
        args_cli.episode_seconds,
        args_cli.min_pelvis_height,
    )
    args_cli.output.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(args_cli.output.resolve())
    client, env = None, None
    with (args_cli.output / "trace.jsonl").open("x") as trace:

        def log(event, **values):
            trace.write(json.dumps({"event": event, **values}, allow_nan=False, default=str) + "\n")
            trace.flush()

        try:
            if args_cli.reference_source == "dp":
                client = DPHttpClient(args_cli.dp_url, args_cli.dp_timeout)
            (args_cli.output / "run_config.json").write_text(
                json.dumps(
                    {
                        "args": vars(args_cli),
                        "env": env_cfg.to_dict(),
                        "agent": agent_cfg.to_dict(),
                        "dp_health": client.health if client else None,
                        "adjustments": adjustments,
                        "warning": "Original motion rewards are diagnostic only, not DP task success.",
                    },
                    indent=2,
                    default=str,
                )
            )
            env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
            raw = env.unwrapped
            if client:
                check_environment(raw, client.protocol)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(Path(args_cli.checkpoint).resolve()), load_optimizer=False, map_location=raw.device)
            policy = runner.get_inference_policy(device=raw.device)
            policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
            command = raw.command_manager.get_term("motion")
            robot, camera = raw.scene["robot"], raw.scene["head_camera"]
            buffer = DPReferenceBuffer(args_cli.replan_steps)
            obs, _ = env.reset()
            policy_nn.reset()
            episode, control_step = 0, 0

            def begin_episode():
                buffer.reset()
                # Render/update without stepping physics; discard previous episode's image.
                raw.sim.render()
                camera.update(0.0, force_recompute=True)
                initial = command.robot_anchor_quat_w.detach().clone()
                if client:
                    client.reset(episode, args_cli.seed + episode)
                log("reset", episode=episode, clip_id=int(command.env_clip[0]))
                return initial

            initial_quat = begin_episode()
            with torch.inference_mode():
                for total_step in range(args_cli.max_steps):
                    if not simulation_app.is_running():
                        break
                    position = command.robot_anchor_pos_w[0].detach().clone()
                    quaternion = command.robot_anchor_quat_w[0].detach().clone()
                    if buffer.needs_prediction(control_step):
                        if client:
                            rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().copy()
                            state = (
                                encode_state(
                                    command.robot_anchor_quat_w,
                                    robot.data.joint_pos,
                                    robot.data.joint_vel,
                                    initial_quat,
                                    client.protocol["active_joint_indices"],
                                )[0]
                                .cpu()
                                .numpy()
                            )
                            start = time.perf_counter()
                            chunk, details = client.infer(rgb, state, control_step)
                            log("prediction", **details, roundtrip_ms=(time.perf_counter() - start) * 1000)
                        else:
                            chunk = replay_reference_chunk(command)
                        buffer.install(chunk, position, quaternion, control_step)
                    groups = buffer.reference_groups(position, quaternion, control_step)
                    if args_cli.reference_source == "motion":
                        for key, value in groups.items():
                            error = float((value - obs[key]).abs().max())
                            torch.testing.assert_close(value, obs[key], atol=2e-5, rtol=2e-5)
                            log(
                                "reference_parity",
                                episode=episode,
                                control_step=control_step,
                                group=key,
                                max_error=error,
                            )
                    actor_obs = obs.clone()
                    for key, value in groups.items():
                        if value.shape != obs[key].shape:
                            raise ValueError(f"PPO reference group mismatch: {key}")
                        actor_obs[key] = value
                    for key in agent_cfg.obs_groups["policy"]:
                        if not torch.isfinite(actor_obs[key]).all():
                            raise FloatingPointError(f"Nonfinite PPO input {key}")
                    actions = policy(actor_obs)
                    if actions.shape != (1, 41) or not torch.isfinite(actions).all():
                        raise FloatingPointError("Invalid PPO action; physics step cancelled")
                    obs, _, dones, _ = env.step(actions)
                    policy_nn.reset(dones)
                    log(
                        "step",
                        episode=episode,
                        control_step=control_step,
                        total_step=total_step,
                        action_abs_max=float(actions.abs().max()),
                        done=bool(dones[0]),
                    )
                    control_step += 1
                    if bool(dones[0]):
                        causes = [
                            name
                            for name in raw.termination_manager.active_terms
                            if bool(raw.termination_manager.get_term(name)[0])
                        ]
                        log("episode_end", episode=episode, causes=causes)
                        episode += 1
                        if episode >= args_cli.episodes:
                            break
                        control_step = 0
                        initial_quat = begin_episode()  # env.step has already auto-reset the scene.
            log("finished", completed_episodes=episode)
            print(f"[INFO] DP rollout finished; trace: {args_cli.output / 'trace.jsonl'}", flush=True)
        except Exception as exc:
            log("error", type=type(exc).__name__, message=str(exc))
            raise
        finally:
            if client:
                try:
                    client.close()
                except Exception as exc:
                    print(f"[WARN] DP session close failed; restart server before next run: {exc}")
            if env:
                env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit shutdown can terminate Python before the normal exception hook runs.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
