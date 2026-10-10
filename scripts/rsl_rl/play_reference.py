"""DINOv2 + reference-flow policy -> existing WEAVE PPO, synchronously."""

import argparse
import json
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
parser.add_argument("--reference-checkpoint", type=Path)
parser.add_argument("--reference-file", type=Path, help="Scene/robot initialization NPZ")
parser.add_argument("--clip-id", type=int, default=0)
parser.add_argument("--camera-config", type=Path, help="Required only if checkpoint lacks recorded calibration")
parser.add_argument("--ppo-config", type=Path)
parser.add_argument("--reference-device", default=None)
parser.add_argument("--reference-source", choices=["policy", "motion"], default="policy")
parser.add_argument("--replan-steps", type=int, default=1, help="1 avoids coarse-grid interpolation")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--episode-seconds", type=float, default=20.0)
parser.add_argument("--max-steps", type=int, default=1000)
parser.add_argument("--min-pelvis-height", type=float, default=0.25)
parser.add_argument("--dino-repo")
parser.add_argument("--dino-weights")
parser.add_argument("--dino-ref", default="main")
parser.add_argument("--output", type=Path)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
# AppLauncher probes this parser before adding its own options. Enforce required
# paths here so that --help also works during that initial probe.
if any(value is None for value in (args.reference_checkpoint, args.reference_file, args.output)):
    parser.error("Specify --reference-checkpoint, --reference-file and --output")
if args.num_envs != 1 or min(args.episodes, args.max_steps, args.replan_steps) <= 0:
    parser.error("One environment and positive episode/step counts are required")
if args.clip_id < 0 or not 0 <= args.seed < 2**32:
    parser.error("Invalid clip ID or seed")
if any(not math.isfinite(x) or x <= 0 for x in (args.episode_seconds, args.min_pelvis_height)):
    parser.error("Invalid episode duration or pelvis height")
if not args.checkpoint or not Path(args.checkpoint).is_file() or not args.reference_file.is_file():
    parser.error("Specify an existing PPO checkpoint and initialization NPZ")
if not args.reference_checkpoint.is_dir() or args.output.exists():
    parser.error("Reference checkpoint must exist and output must be new")
args.enable_cameras = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/weave_policy"))
sys.argv = [sys.argv[0]] + hydra_args
launcher = AppLauncher(args)
simulation_app = launcher.app

import time

import g1_hoi_learning.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import yaml
from g1_hoi_learning.rollout.reference_play_config import configure_reference_play
from rsl_rl.runners import OnPolicyRunner
from weave_policy.config import ReferencePolicyConfig
from weave_policy.data.reference_codec import ReferenceBuffer
from weave_policy.predictor import ReferencePredictor
from weave_policy.weave_adapter import check_environment, replay_reference

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    config = ReferencePolicyConfig.from_dict(json.loads((args.reference_checkpoint / "config.json").read_text()))
    metadata = json.loads((args.reference_checkpoint / "metadata.json").read_text())
    recorded_camera = metadata["source_info"].get("camera_calibration")
    if args.camera_config is not None:
        supplied = json.loads(args.camera_config.read_text())["env"]["scene"]["head_camera"]
        if recorded_camera is not None and supplied != recorded_camera:
            raise ValueError("Camera configuration differs from training calibration")
        recorded_camera = supplied
    if recorded_camera is None:
        raise ValueError("Checkpoint has no camera calibration; specify collection --camera-config")
    ppo_config = args.ppo_config or Path(args.checkpoint).resolve().parent / "params/agent.yaml"
    saved = yaml.safe_load(ppo_config.read_text())
    if saved.get("class_name") != "OnPolicyRunner":
        raise ValueError("Expected original WEAVE OnPolicyRunner configuration")
    agent_cfg.from_dict(saved)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    if args.device is not None:
        env_cfg.sim.device = args.device
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.policy.compile = False
    env_cfg.seed = args.seed
    with np.load(args.reference_file, allow_pickle=True) as motion:
        if len(set(map(str, motion["object_names"]))) != 1 or args.clip_id >= len(motion["motion_lengths"]):
            raise ValueError("Expected one object and a valid initialization clip")
        if abs(float(np.asarray(motion["fps"]).reshape(-1)[0]) - config.fps) > 1e-6:
            raise ValueError("Initialization motion FPS mismatch")
    adjustments = configure_reference_play(
        env_cfg, args.reference_file, args.clip_id, recorded_camera, args.episode_seconds, args.min_pelvis_height
    )
    buffer = ReferenceBuffer(config.reference_offsets, args.replan_steps)
    predictor = None
    if args.reference_source == "policy":
        predictor = ReferencePredictor(args.reference_checkpoint, args.reference_device or env_cfg.sim.device)
        predictor.load_encoder(args.dino_repo, args.dino_weights, args.dino_ref)
    args.output.mkdir(parents=True)
    env_cfg.log_dir = str(args.output.resolve())
    run = {
        "args": vars(args),
        "model": config.to_dict(),
        "env": env_cfg.to_dict(),
        "agent": agent_cfg.to_dict(),
        "adjustments": adjustments,
    }
    (args.output / "run_config.json").write_text(json.dumps(run, indent=2, default=str))
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        raw = env.unwrapped
        check_environment(raw, config, metadata["source_info"])
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint, load_optimizer=False, map_location=raw.device)
        ppo = runner.get_inference_policy(device=raw.device)
        ppo_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        command = raw.command_manager.get_term("motion")
        camera = raw.scene["head_camera"]
        obs, _ = env.reset()
        ppo_nn.reset()
        episode, step = 0, 0

        def begin_episode():
            buffer.reset()
            raw.sim.render()
            camera.update(0.0, force_recompute=True)

        begin_episode()
        with (args.output / "trace.jsonl").open("x") as trace, torch.inference_mode():
            for total in range(args.max_steps):
                if not simulation_app.is_running():
                    break
                pos = command.robot_anchor_pos_w[0].clone()
                quat = command.robot_anchor_quat_w[0].clone()
                prediction_ms = None
                if buffer.needs_prediction(step):
                    started = time.perf_counter()
                    if predictor is None:
                        chunk = replay_reference(command, config.reference_offsets)
                    else:
                        rgb = camera.data.output["rgb"][..., :3]
                        chunk = predictor.predict_rgb(rgb, seed=args.seed + episode * args.max_steps + step)[0]
                        chunk = chunk.to(raw.device)
                    prediction_ms = (time.perf_counter() - started) * 1000
                    buffer.install(chunk, pos, quat, step)
                groups = buffer.groups(pos, quat, step)
                if predictor is None and args.replan_steps == 1:
                    for key, value in groups.items():
                        torch.testing.assert_close(value, obs[key], atol=2e-5, rtol=2e-5)
                actor_obs = obs.clone()
                for key, value in groups.items():
                    if value.shape != actor_obs[key].shape:
                        raise ValueError(f"Reference shape mismatch for {key}")
                    actor_obs[key] = value
                if any(not torch.isfinite(actor_obs[key]).all() for key in agent_cfg.obs_groups["policy"]):
                    raise FloatingPointError("Nonfinite PPO observation")
                actions = ppo(actor_obs)
                if actions.shape != (1, 41) or not torch.isfinite(actions).all():
                    raise FloatingPointError("Invalid PPO action")
                obs, _, dones, _ = env.step(actions)
                ppo_nn.reset(dones)
                trace.write(
                    json.dumps(
                        {
                            "episode": episode,
                            "step": step,
                            "total_step": total,
                            "prediction_ms": prediction_ms,
                            "done": bool(dones[0]),
                            "action_abs_max": float(actions.abs().max()),
                        }
                    )
                    + "\n"
                )
                trace.flush()
                step += 1
                if bool(dones[0]):
                    episode += 1
                    if episode >= args.episodes:
                        break
                    step = 0
                    begin_episode()
        print(f"Reference rollout trace: {args.output / 'trace.jsonl'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
