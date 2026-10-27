"""Exercise eval's recording lifecycle with real Gymnasium/MP4 output, without Isaac Sim."""

import ast
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm


@pytest.mark.parametrize(
    "video,limit,interrupt,expected_frames",
    [(False, 0, False, 0), (True, 0, False, 3), (True, 1, False, 1), (True, 0, True, 1)],
)
def test_eval_video_lifecycle(tmp_path, video, limit, interrupt, expected_frames):
    class CameraEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
        device, num_envs, step_dt = "cpu", 1, 0.02

        def __init__(self):
            self.render_mode = "rgb_array" if video else None
            self.steps, self.resets, self.renders = 0, 0, 0
            self.closed = False

        def reset(self, **kwargs):
            self.resets += 1
            return torch.zeros(1), {}

        def step(self, action):
            if interrupt and self.steps == 1:
                raise KeyboardInterrupt
            self.steps += 1
            return torch.zeros(1), 0.0, False, False, {"time_outs": torch.tensor([False])}

        def render(self):
            self.renders += 1
            assert self.resets == 2  # Initialization resets must not be recorded.
            return np.full((32, 32, 3), self.steps * 40, dtype=np.uint8)

        def close(self):
            self.closed = True

    raw = CameraEnv()

    class VecWrapper:
        def __init__(self, env, **kwargs):
            self.env, self.unwrapped, self.num_envs = env, env.unwrapped, 1
            env.reset()  # RslRlVecEnvWrapper also resets during construction.

        def reset(self):
            return self.env.reset()

        def step(self, actions):
            obs, reward, _, _, extras = self.env.step(actions)
            return obs, reward, torch.tensor([raw.steps >= 3]), extras

        def close(self):
            self.env.close()

    class Runner:
        def __init__(self, *args, **kwargs):
            self.alg = SimpleNamespace(policy=SimpleNamespace(reset=lambda *args: None))

        def load(self, path):
            pass

        def get_inference_policy(self, **kwargs):
            return lambda obs: torch.zeros(1)

    class Evaluator:
        max_steps = 5

        def __init__(self, *args):
            self.finished = self.success = torch.tensor([False])

        def collect(self):
            pass

        def update(self, dones, time_outs):
            self.finished = self.success = dones

        @property
        def done(self):
            return bool(self.finished.all())

        def results(self):
            return {"steps": raw.steps}

    args = SimpleNamespace(
        device=None, num_envs=1, checkpoint=str(tmp_path / "model.pt"),
        output_dir=str(tmp_path / "eval"), task="test", video=video,
        video_length=limit, real_time=False,
    )
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(device="cpu"),
        commands=SimpleNamespace(motion=SimpleNamespace(
            eval_success_mode="reference", motion_files=[], eval_start_mode="zero", eval_clip_mode="same_clip"
        )),
        terminations=SimpleNamespace(), scene=SimpleNamespace(),
    )
    agent_cfg = SimpleNamespace(
        seed=0, experiment_name="test", clip_actions=None, class_name="OnPolicyRunner",
        device="cpu", to_dict=lambda: {},
    )
    ns = dict(
        gym=SimpleNamespace(make=lambda *a, **kw: raw, wrappers=gym.wrappers),
        torch=torch, os=os, time=time, json=json, tqdm=tqdm, args_cli=args,
        cli_args=SimpleNamespace(update_rsl_rl_cfg=lambda cfg, args: cfg),
        DoneTerm=lambda **kwargs: kwargs, motion_clip_end=None,
        read_motion_num_clips=lambda files: 1, read_motion_clip_names=lambda files: ["clip0"],
        retrieve_file_path=lambda path: path, DirectMARLEnv=type("DirectMARLEnv", (), {}),
        RslRlVecEnvWrapper=VecWrapper, OnPolicyRunner=Runner, ClipEvaluator=Evaluator,
        simulation_app=SimpleNamespace(is_running=lambda: True), print_report=lambda res: None,
    )
    path = Path(__file__).resolve().parents[3] / "scripts/rsl_rl/eval.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            ns["main"](env_cfg, agent_cfg)
    else:
        ns["main"](env_cfg, agent_cfg)
        result = json.loads((tmp_path / "eval/metrics.json").read_text())
        assert result["steps"] == 3
        assert result["eval_start_mode"] == "zero"
        assert result["eval_clip_mode"] == "same_clip"
        assert result["seed"] == 0
    assert raw.closed
    assert raw.renders == expected_frames
    video_path = tmp_path / "eval/videos/eval.mp4"
    if video:
        assert video_path.is_file()
        with VideoFileClip(str(video_path)) as clip:
            assert clip.fps == 50
            assert round(clip.duration * clip.fps) == expected_frames
    else:
        assert not video_path.exists()
