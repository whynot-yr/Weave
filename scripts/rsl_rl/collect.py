# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect synchronized WEAVE policy rollouts with head-camera RGB-D."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Collect structured WEAVE policy rollouts.")
parser.add_argument("--task", type=str, default="G1-Inspire-HOI-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None, help="Environment seed; defaults to the runner seed.")
parser.add_argument(
    "--reference_file",
    type=str,
    default="/data/g1_hoi_ws/Weave/data/train/floorlamp_805clip.npz",
)
parser.add_argument("--num_rollouts", type=int, default=100)
parser.add_argument("--sampling_seed", type=int, default=42)
parser.add_argument("--contact_threshold", type=float, default=1.0)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory. Defaults to ./datasets/floorlamp_rollout_<num_rollouts>_seed<sampling_seed>.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim starts."""

import json

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import g1_hoi_learning.tasks  # noqa: F401
from g1_hoi_learning.rollout import RolloutRecorder
from g1_hoi_learning.tasks.hoi.mdp.terminations import motion_clip_end


def load_reference_index(path: str) -> tuple[np.ndarray, list[str], list[str], int]:
    with np.load(path, allow_pickle=True) as data:
        lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
        names = [str(x) for x in np.asarray(data["motion_names"])]
        objects = [str(x) for x in np.asarray(data["object_names"])]
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    if not (len(lengths) == len(names) == len(objects)):
        raise ValueError("motion_lengths, motion_names and object_names must have equal lengths")
    return lengths, names, objects, fps


def resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.isfile(args_cli.checkpoint):
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    reference_file = os.path.abspath(args_cli.reference_file)
    default_output = f"./datasets/floorlamp_rollout_{args_cli.num_rollouts}_seed{args_cli.sampling_seed}"
    output_dir = os.path.abspath(args_cli.output_dir or default_output)
    lengths, clip_names, object_names, reference_fps = load_reference_index(reference_file)
    if len(set(object_names)) != 1:
        raise ValueError(f"V1 collection requires one object, found {sorted(set(object_names))}")
    if args_cli.num_rollouts <= 0 or args_cli.num_rollouts > len(lengths):
        raise ValueError(f"num_rollouts must be in [1, {len(lengths)}]")
    rng = np.random.default_rng(args_cli.sampling_seed)
    selected_ids = rng.choice(len(lengths), size=args_cli.num_rollouts, replace=False).astype(np.int64)

    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_rollouts
    env_cfg.commands.motion.motion_files = [reference_file]
    env_cfg.commands.motion.eval_mode = True
    env_cfg.commands.motion.rsi = False
    env_cfg.commands.motion.eval_clip_ids = selected_ids.tolist()
    env_cfg.terminations.clip_end = DoneTerm(func=motion_clip_end, params={"command_name": "motion"}, time_out=True)
    env_cfg.num_rerenders_on_reset = 1

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    if abs(reference_fps - 1.0 / control_dt) > 1.0e-6:
        raise ValueError(f"reference is {reference_fps} Hz but environment control is {1.0 / control_dt:g} Hz")

    checkpoint = resolve_checkpoint(agent_cfg)
    log_dir = os.path.dirname(checkpoint)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Reference: {reference_file}")
    print(f"[INFO] Selected {len(selected_ids)}/{len(lengths)} unique clips with seed {args_cli.sampling_seed}")
    print(f"[INFO] Clip ids: {selected_ids.tolist()}")
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    if any(os.scandir(output_dir)):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic

    obs, _ = env.reset()
    policy_nn.reset()
    recorder = RolloutRecorder(
        env,
        output_dir,
        initial_obs=obs,
        clip_names=clip_names,
        selection_seed=args_cli.sampling_seed,
        environment_seed=env_cfg.seed,
        reference_file=reference_file,
        checkpoint=checkpoint,
        contact_threshold=args_cli.contact_threshold,
    )
    with open(os.path.join(output_dir, "selection.json"), "w", encoding="utf-8") as stream:
        json.dump(
            {
                "reference_file": reference_file,
                "sampling_seed": args_cli.sampling_seed,
                "sampling_strategy": "uniform_without_replacement",
                "selected_clip_ids": selected_ids.tolist(),
                "selected_clip_names": [clip_names[i] for i in selected_ids],
            },
            stream,
            indent=2,
        )
    with open(os.path.join(output_dir, "resolved_config.json"), "w", encoding="utf-8") as stream:
        json.dump(
            {"env": env_cfg.to_dict(), "agent": agent_cfg.to_dict()},
            stream,
            indent=2,
            default=str,
        )

    pbar = tqdm(range(recorder.max_steps), desc="collect", unit="step", dynamic_ncols=True)
    try:
        for _ in pbar:
            if not simulation_app.is_running():
                print("[WARN] Simulation closed before all rollouts finished.")
                break
            with torch.inference_mode():
                actions = policy(obs)
                pending = recorder.capture_before_step(obs, actions)
                obs, rewards, dones, _ = env.step(actions)
                recorder.capture_after_step(pending, rewards)
                policy_nn.reset(dones)
            finished = int((~recorder.active).sum())
            succeeded = int(recorder.success.sum())
            pbar.set_postfix_str(f"done {finished}/{env.num_envs} | success {succeeded}")
            if recorder.done:
                break
    finally:
        pbar.close()
        recorder.close()
        env.close()

    print(
        f"[INFO] Collected {recorder.num_envs} episodes, {int(recorder.success.sum())} successful, "
        f"{int(recorder.lengths.sum())} frames"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
