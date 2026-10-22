"""Goal lifecycle and real RSL-RL runner checks without starting Isaac Sim."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorClass, TensorDict

PACKAGE = Path(__file__).resolve().parents[1] / "g1_hoi_learning"


def definitions(path, names, namespace):
    """Load simulator-dependent definitions with a minimal environment test double."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def command():
    ns = definitions(
        PACKAGE / "tasks/hoi/mdp/commands.py",
        {"MotionData", "MotionLoader", "MotionCommand"},
        dict(torch=torch, np=np, Sequence=Sequence, TensorClass=TensorClass, CommandTerm=object),
    )
    loader = ns["MotionLoader"].__new__(ns["MotionLoader"])
    loader.clip_lengths = torch.tensor([4, 6, 3])
    loader.clip_starts = torch.tensor([0, 4, 10, 13])
    loader.first_clip = lambda objects: torch.zeros_like(objects)
    loader.sample_clip = lambda objects: torch.ones_like(objects)
    frame = torch.arange(13).float()
    quats = torch.tensor([1.0, 0, 0, 0]).expand(13, 2, 4).clone()
    loader.buffer = ns["MotionData"](
        motion_id=torch.repeat_interleave(torch.arange(3), loader.clip_lengths),
        step=torch.cat([torch.arange(length) for length in loader.clip_lengths]),
        joint_pos=frame[:, None].expand(-1, 3).clone() / 100,
        joint_vel=torch.zeros(13, 3),
        body_pos_w=frame[:, None, None].expand(-1, 2, 3).clone(),
        body_quat_w=quats,
        body_lin_vel_w=torch.zeros(13, 2, 3),
        body_ang_vel_w=torch.zeros(13, 2, 3),
        object_pos_w=frame[:, None].expand(-1, 3).clone() + 0.5,
        object_quat_w=quats[:, 0].clone(),
        object_lin_vel_w=torch.zeros(13, 3),
        object_ang_vel_w=torch.zeros(13, 3),
        contact_label=torch.ones(13, 2),
        batch_size=[13],
    )
    cmd = ns["MotionCommand"].__new__(ns["MotionCommand"])
    cmd.motion, cmd.num_envs, cmd.device, cmd.anchor_index = loader, 3, "cpu", 0
    cmd.body_indices = [0, 1]
    cmd.env_object, cmd.env_clip = torch.zeros(3, dtype=torch.long), torch.tensor([0, 2, 1])
    cmd.time_steps, cmd.future_offsets = torch.tensor([1, 0, 3]), torch.tensor([0, 2])
    cmd.cfg = SimpleNamespace(eval_mode=False, rsi=True)
    origins = torch.tensor([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 100.0]])
    cmd._env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    actual = SimpleNamespace(
        body_pos_w=origins[:, None].expand(-1, 2, -1).clone(),
        body_quat_w=quats[:3].clone(),
        soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).expand(3, 3, 2),
    )

    def write_root(state, env_ids):
        actual.body_pos_w[env_ids, 0] = state[:, :3]
        actual.body_quat_w[env_ids, 0] = state[:, 3:7]

    cmd.robot = SimpleNamespace(
        data=actual, write_root_state_to_sim=write_root, write_joint_state_to_sim=lambda *args, **kwargs: None
    )
    cmd.object = SimpleNamespace(write_root_state_to_sim=lambda *args, **kwargs: None)
    cmd._refresh_goal()
    cmd._refresh_caches()
    return cmd


def test_terminal_goal_ignores_rsi_and_does_not_alias_motion(command):
    torch.testing.assert_close(command.goal_joint_pos[:, 0], torch.tensor([0.03, 0.12, 0.09]))
    before = command.goal_obj_pos_w.clone()
    command.time_steps += 1
    command._refresh_caches()
    torch.testing.assert_close(command.goal_obj_pos_w, before)
    torch.testing.assert_close(command.joint_pos[:, 0], torch.tensor([0.02, 0.11, 0.08]))
    command._goal_joint_pos.zero_()
    assert command.motion.buffer.joint_pos[3, 0].item() == pytest.approx(0.03)


def test_partial_reset_and_rollover_refresh_goals_and_reference(command):
    before = command.goal_joint_pos.clone()
    command._resample_command([0])
    assert command.env_clip.tolist() == [1, 2, 1]
    torch.testing.assert_close(command.goal_joint_pos[0], torch.full((3,), 0.09))
    torch.testing.assert_close(command.goal_joint_pos[1:], before[1:])
    # The first observation after an explicit reset must use the newly selected clip/t.
    torch.testing.assert_close(command.joint_pos, command.motion.frames(command.env_clip, command.time_steps).joint_pos)
    command.motion.sample_clip = lambda objects: torch.zeros_like(objects)
    command.time_steps = torch.tensor([5, 0, 0])
    command._update_command()
    assert command.env_clip.tolist() == [0, 2, 1]
    torch.testing.assert_close(command.goal_joint_pos[0], torch.full((3,), 0.03))
    torch.testing.assert_close(command.goal_joint_pos[1:], before[1:])


def test_eval_reset_preserves_frame_zero_reference(command):
    command.cfg.eval_mode = True
    command._eval_env_clip = torch.tensor([1, 0, 2])
    command._resample_command(torch.arange(3))
    assert command.time_steps.tolist() == [0, 0, 0]
    torch.testing.assert_close(command.joint_pos[:, 0], torch.tensor([0.04, 0.0, 0.10]))
    torch.testing.assert_close(command.goal_joint_pos[:, 0], torch.tensor([0.09, 0.03, 0.12]))


def test_goal_observations_reanchor_world_target(command):
    # Prefer the installed editable package over the repository's isaaclab/ namespace.
    specs = (finder.find_spec("isaaclab", None) for finder in sys.meta_path)
    package_spec = next(spec for spec in specs if spec is not None and spec.origin is not None)
    isaac_root = Path(package_spec.origin).parent
    spec = importlib.util.spec_from_file_location("goal_test_math", isaac_root / "utils/math.py")
    math_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_module)
    ns = definitions(
        PACKAGE / "tasks/hoi/mdp/observations.py",
        {"goal_robot_pose", "goal_object_pose"},
        dict(
            torch=torch,
            matrix_from_quat=math_module.matrix_from_quat,
            subtract_frame_transforms=math_module.subtract_frame_transforms,
        ),
    )
    env = SimpleNamespace(num_envs=3, command_manager=SimpleNamespace(get_term=lambda name: command))
    robot, obj = ns["goal_robot_pose"](env, "motion"), ns["goal_object_pose"](env, "motion")
    assert robot.shape == (3, 12) and obj.shape == (3, 9)
    torch.testing.assert_close(robot[:, :3], command.goal_joint_pos)
    torch.testing.assert_close(obj[0], torch.tensor([3.5, 3.5, 3.5, 1.0, 0, 0, 1, 0, 0]))
    world_goal = command.goal_obj_pos_w.clone()
    command.robot.data.body_pos_w[:, 0, 0] += 1
    half = torch.tensor(torch.pi / 4)
    command.robot.data.body_quat_w[:, 0] = torch.tensor([half.cos(), 0.0, 0.0, half.sin()])
    changed = ns["goal_object_pose"](env, "motion")
    torch.testing.assert_close(changed[0], torch.tensor([3.5, -2.5, 3.5, 0, 1.0, -1.0, 0, 0, 0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(command.goal_obj_pos_w, world_goal)


@pytest.fixture
def learning(monkeypatch):
    # Import actual learning modules without the package initializer's simulator imports.
    for suffix in ("", ".algorithms", ".algorithms.ppo", ".algorithms.distillation"):
        name = "g1_hoi_learning" + suffix
        module = ModuleType(name)
        module.__path__ = [str(PACKAGE.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(PACKAGE)]
        monkeypatch.setitem(sys.modules, name, module)

    def load(suffix):
        name = "g1_hoi_learning." + suffix
        spec = importlib.util.spec_from_file_location(name, PACKAGE.joinpath(*suffix.split(".")).with_suffix(".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("algorithms.networks")
    policy_module = load("algorithms.ppo.networks")
    load("algorithms.optimizers")
    load("algorithms.ppo.algorithm")
    distill = load("algorithms.distillation.networks")
    algorithm = load("algorithms.distillation.algorithm")
    import rsl_rl.runners.on_policy_runner as runner_module

    monkeypatch.setattr(runner_module, "SimBaActorCriticTeacher", distill.SimBaActorCriticTeacher, raising=False)
    monkeypatch.setattr(runner_module, "MuonPPODistill", algorithm.MuonPPODistill, raising=False)
    groups = {
        "policy": ["goal_robot_pose", "goal_object_pose", "object_state", "robot_proprio"],
        "teacher": ["ref_motion_body", "ref_motion_object", "object_state", "robot_proprio"],
        "critic": ["ref_motion_body", "ref_motion_object", "object_state", "robot_privileged"],
    }
    dims = dict(
        goal_robot_pose=12,
        goal_object_pose=9,
        object_state=5,
        robot_proprio=4,
        ref_motion_body=8,
        ref_motion_object=6,
        robot_privileged=3,
    )
    obs = TensorDict({key: torch.randn(4, dim) for key, dim in dims.items()}, batch_size=[4])
    policy_cfg = dict(
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        expansion=1,
        latent_dim=4,
        compile=False,
        init_noise_std=0.2,
        encoder_hidden_dims={g: {"type": "mlp", "hidden_dims": [8]} for g in dims},
    )
    tracker = policy_module.SimBaActorCritic(
        obs, {"policy": groups["teacher"], "critic": groups["critic"]}, 2, **policy_cfg
    )
    return SimpleNamespace(
        runner=runner_module.OnPolicyRunner,
        tracker=tracker,
        obs=obs,
        groups=groups,
        policy_cfg=policy_cfg,
        distill=distill,
    )


def test_shared_encoders_transfer_by_name(learning):
    groups = dict(learning.groups, policy=["robot_proprio", "goal_robot_pose", "object_state", "goal_object_pose"])
    policy = learning.distill.SimBaActorCriticTeacher(
        learning.obs, groups, 2, teacher_hidden_dim=8, teacher_num_blocks=1, **learning.policy_cfg
    )
    goal_before = {k: v.clone() for k, v in policy.actor[0].encoders[1].state_dict().items()}
    assert policy.load_state_dict(learning.tracker.state_dict()) is False
    for student_index, teacher_index in ((0, 3), (2, 2)):
        for left, right in zip(
            policy.actor[0].encoders[student_index].parameters(),
            learning.tracker.actor[0].encoders[teacher_index].parameters(),
            strict=True,
        ):
            torch.testing.assert_close(left, right)
    for key, value in goal_before.items():
        torch.testing.assert_close(policy.actor[0].encoders[1].state_dict()[key], value)


def test_standard_runner_two_updates_and_termination_logging(learning, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    class Env:
        num_envs, num_actions, device = 4, 2, "cpu"
        steps = 0

        def get_observations(self):
            return learning.obs.clone()

        def step(self, actions):
            assert actions.shape == (4, 2) and torch.isfinite(actions).all()
            self.steps += 1
            obs = self.get_observations()
            dones = torch.full((4,), self.steps % 2 == 0, dtype=torch.long)
            return obs, torch.ones(4), dones, {"log": {"Episode_Termination/time_out": 0.5}}

    cfg = dict(
        num_steps_per_env=2,
        save_interval=100,
        obs_groups=learning.groups,
        policy=dict(
            learning.policy_cfg, class_name="SimBaActorCriticTeacher", teacher_hidden_dim=8, teacher_num_blocks=1
        ),
        algorithm=dict(
            class_name="MuonPPODistill",
            num_learning_epochs=1,
            num_mini_batches=2,
            bc_coef=0.5,
            bc_coef_anneal_iters=10,
            learning_rate=1e-4,
        ),
    )
    runner = learning.runner(Env(), copy.deepcopy(cfg), log_dir=str(tmp_path))
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": learning.tracker.state_dict(), "infos": None}, teacher_path)
    runner.load(str(teacher_path))
    teacher_before = {k: v.clone() for k, v in runner.alg.policy.teacher.state_dict().items()}
    actor_before = [p.detach().clone() for p in runner.alg.policy.actor.parameters()]
    runner.learn(num_learning_iterations=2)
    runner.writer.close()
    assert runner.env.steps == 4 and runner.alg._it == 2 and runner.alg.storage.step == 0
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, runner.alg.policy.actor.parameters(), strict=True))
    for key, value in teacher_before.items():
        torch.testing.assert_close(runner.alg.policy.teacher.state_dict()[key], value)
    events = EventAccumulator(str(tmp_path)).Reload()
    assert "Episode_Termination/time_out" in events.Tags()["scalars"]
    assert (tmp_path / "model_1.pt").is_file()
    restored = learning.runner(Env(), copy.deepcopy(cfg), log_dir=str(tmp_path / "resume"))
    restored.load(str(tmp_path / "model_1.pt"))
    with torch.no_grad():
        torch.testing.assert_close(
            restored.alg.policy.act_inference(learning.obs), runner.alg.policy.act_inference(learning.obs)
        )
