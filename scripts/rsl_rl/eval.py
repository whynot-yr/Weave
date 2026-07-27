# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a checkpoint if an RL agent from RSL-RL.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluation of an RSL-RL agent.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments. Defaults to the number of clips."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--output_dir", type=str, default=None, help="Where to write metrics.json. Defaults to <run>/eval.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import time

import gymnasium as gym
import torch
import numpy as np
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tqdm import tqdm

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import g1_hoi_learning.tasks  # noqa: F401
from g1_hoi_learning.tasks.hoi.mdp.commands import MotionCommand
from g1_hoi_learning.tasks.hoi.mdp.terminations import motion_clip_end


def read_motion_num_clips(motion_files: list[str]) -> int:
    return sum(len(np.asarray(np.load(f, allow_pickle=True)["motion_lengths"])) for f in motion_files)


def read_motion_clip_names(motion_files: list[str]) -> list[str]:
    return [str(x) for f in motion_files for x in np.asarray(np.load(f, allow_pickle=True)["motion_names"])]


class ClipEvaluator:
    """Accumulates per-clip statistics over each env's first episode.
    """

    def __init__(self, env: RslRlVecEnvWrapper, clip_names: list[str], command_name: str = "motion"):
        self.env = env.unwrapped
        self.device = self.env.device
        self.command: MotionCommand = self.env.command_manager.get_term(command_name)
        self.term_mgr = self.env.termination_manager
        self.term_names = list(self.term_mgr.active_terms)
        self.fail_names = [n for n in self.term_names if not self.term_mgr.get_term_cfg(n).time_out]
        if "clip_end" not in self.term_names:
            raise ValueError("clip-wise eval needs the 'clip_end' termination term; it is not active.")

        self.metric_keys = list(self.command.metrics)

        n = self.env.num_envs
        self.clip_ids = self.command.env_clip.clone()
        self.clip_len = self.command.motion.clip_lengths[self.clip_ids].clone()
        self.clip_names = [clip_names[int(i)] for i in self.clip_ids]

        self.finished = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cause = [""] * n
        self.sums = {k: torch.zeros(n, device=self.device) for k in self.metric_keys}

    @property
    def max_steps(self) -> int:
        return int(self.clip_len.max().item()) + 5

    def collect(self) -> None:
        self.command._update_metrics()
        active = (~self.finished).float()
        for k in self.metric_keys:
            self.sums[k] += self.command.metrics[k] * active
        self.steps += ~self.finished

    def update(self, dones: torch.Tensor, time_outs: torch.Tensor) -> None:
        newly = dones.bool() & ~self.finished
        if not newly.any():
            return
        terminated = dones.bool() & ~time_outs
        flags = {n: self.term_mgr.get_term(n) for n in self.term_names}
        self.success |= newly & flags["clip_end"] & ~terminated
        for i in newly.nonzero(as_tuple=False).flatten().tolist():
            if terminated[i]:
                self.cause[i] = next((n for n in self.fail_names if flags[n][i]), "unknown")
            else:
                self.cause[i] = "clip_end" if flags["clip_end"][i] else "time_out"
        self.finished |= newly

    @property
    def done(self) -> bool:
        return bool(self.finished.all())

    def results(self) -> dict:
        steps = self.steps.clamp(min=1)
        per_clip = {k: (v / steps).cpu().tolist() for k, v in self.sums.items()}
        progress = (self.steps.float() / self.clip_len.clamp(min=1).float()).clamp(max=1.0)
        success = self.success.cpu()

        def agg(mask: torch.Tensor) -> dict[str, float]:
            if not mask.any():
                return {k: float("nan") for k in self.metric_keys}
            frames = self.steps[mask].sum().float()
            return {k: float(self.sums[k][mask].sum() / frames) for k in self.metric_keys}

        return {
            "num_clips": int(self.env.num_envs),
            "success_rate": float(success.float().mean()),
            "progress_rate": float(progress.mean()),
            "metric_keys": self.metric_keys,
            "metrics_all": agg(torch.ones_like(self.success)),
            "metrics_success": agg(self.success),
            "failures": {n: self.cause.count(n) for n in self.fail_names if self.cause.count(n) > 0},
            "clips": [
                {
                    "env": i,
                    "clip_id": int(self.clip_ids[i]),
                    "name": self.clip_names[i],
                    "length": int(self.clip_len[i]),
                    "steps": int(self.steps[i]),
                    "progress": float(progress[i]),
                    "success": bool(success[i]),
                    "cause": self.cause[i],
                    **{k: per_clip[k][i] for k in self.metric_keys},
                }
                for i in range(self.env.num_envs)
            ],
        }


def print_report(res: dict) -> None:
    """Aggregate summary only; per-clip rows live in metrics.json."""
    keys = res["metric_keys"]
    width = max(len(k) for k in keys)
    n_ok = sum(c["success"] for c in res["clips"])
    print(f"\n===== CLIP-WISE EVALUATION  ({res['num_clips']} clips) =====")
    print(f"success_rate : {res['success_rate']:.4f}  ({n_ok}/{res['num_clips']})")
    print(f"progress_rate: {res['progress_rate']:.4f}")
    print(f"{'metric':<{width}}  {'all':>10}  {'success':>10}")
    for k in keys:
        print(f"{k:<{width}}  {res['metrics_all'][k]:>10.4f}  {res['metrics_success'][k]:>10.4f}")
    if res["failures"]:
        print("failures     : " + "  ".join(f"{k}={v}" for k, v in sorted(res["failures"].items(), key=lambda x: -x[1])))
    print()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL agent clip by clip."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.commands.motion.eval_mode = True
    env_cfg.terminations.clip_end = DoneTerm(func=motion_clip_end, params={"command_name": "motion"}, time_out=True)
    num_clips = read_motion_num_clips(env_cfg.commands.motion.motion_files)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else num_clips
    print(f"[INFO] Clip-wise eval: {num_clips} clip(s) -> {env_cfg.scene.num_envs} env(s)")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic

    dt = env.unwrapped.step_dt
    obs, _ = env.reset()
    policy_nn.reset()
    evaluator = ClipEvaluator(env, read_motion_clip_names(env_cfg.commands.motion.motion_files))

    pbar = tqdm(range(evaluator.max_steps), desc="eval", unit="step", dynamic_ncols=True)
    for _ in pbar:
        if not simulation_app.is_running():
            print("[WARN] Simulation app closed before every clip finished; results are partial.")
            break
        start_time = time.time()
        evaluator.collect()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            policy_nn.reset(dones)
        evaluator.update(dones, extras["time_outs"])
        n_done, n_ok = int(evaluator.finished.sum()), int(evaluator.success.sum())
        pbar.set_postfix_str(f"done {n_done}/{env.num_envs} | ok {n_ok} | succ {n_ok / max(n_done, 1)*100:.2f}%")
        if evaluator.done:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    pbar.close()

    if not evaluator.done:
        print(f"[WARN] {int((~evaluator.finished).sum())} env(s) never terminated; scored on what they ran.")

    res = evaluator.results()
    print_report(res)

    output_dir = args_cli.output_dir or os.path.join(log_dir, "eval")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "metrics.json")
    res["checkpoint"] = resume_path
    res["motion_files"] = list(env_cfg.commands.motion.motion_files)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[INFO] Wrote {out_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
