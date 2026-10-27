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


@pytest.fixture(scope="module")
def isaac_math():
    # Prefer the installed editable package over the repository's isaaclab/ namespace.
    specs = (finder.find_spec("isaaclab", None) for finder in sys.meta_path)
    package_spec = next(spec for spec in specs if spec is not None and spec.origin is not None)
    spec = importlib.util.spec_from_file_location("goal_test_math", Path(package_spec.origin).parent / "utils/math.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def command(isaac_math):
    ns = definitions(
        PACKAGE / "tasks/hoi/mdp/commands.py",
        {"MotionData", "MotionLoader", "MotionCommand"},
        dict(
            torch=torch, np=np, Sequence=Sequence, TensorClass=TensorClass, CommandTerm=object,
            quat_error_magnitude=isaac_math.quat_error_magnitude,
        ),
    )
    loader = ns["MotionLoader"].__new__(ns["MotionLoader"])
    loader.clip_lengths = torch.tensor([4, 6, 3])
    loader.num_clips = 3
    loader.clip_object = torch.zeros(3, dtype=torch.long)
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
    cmd.init_clip_ids, cmd.init_time_steps = cmd.env_clip.clone(), cmd.time_steps.clone()
    cmd.goal_time_steps = torch.zeros(3, dtype=torch.long)
    cmd.first_goal_time_steps = torch.zeros_like(cmd.time_steps)
    cmd.goal_reached = torch.zeros(3, dtype=torch.bool)
    cmd.cfg = SimpleNamespace(
        eval_success_mode="reference", eval_mode=False, eval_start_mode="zero", eval_clip_mode="same_clip", rsi=True, goal_sampling_mode="terminal"
    )
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
    cmd._resample_goal()
    cmd.first_goal_time_steps.copy_(cmd.goal_time_steps)
    cmd._refresh_caches()
    return cmd


def test_goal_visualization_uses_sampled_world_goals_and_filters_envs(command):
    command.cfg.show_goal_poses = True
    command.cfg.show_reference_bodies = False
    command.cfg.goal_vis_env_ids = [2]
    command._vis_env_ids = torch.tensor([2])
    command.robot.is_initialized = True
    command.object.data = SimpleNamespace(
        root_pos_w=torch.zeros(3, 3),
        root_quat_w=command.robot.data.body_quat_w[:, 0],
    )
    drawn = {}

    def recorder(name):
        return SimpleNamespace(visualize=lambda pos, quat: drawn.update({name: (pos.clone(), quat.clone())}))

    names = ("target_root", "target_object", "target_root_dot", "target_object_dot",
             "current_anchor", "current_object")
    command._pose_visualizers = {name: recorder(name) for name in names}
    command._debug_vis_callback(None)
    torch.testing.assert_close(drawn["target_root"][0], command.goal_anchor_pos_w[[2]])
    torch.testing.assert_close(drawn["target_object"][0], command.goal_obj_pos_w[[2]])
    torch.testing.assert_close(drawn["target_object"][1], command.goal_obj_quat_w[[2]])
    assert not torch.equal(drawn["target_root"][0], command.anchor_pos_w[[2]])

    # A new sampled frame must replace both displayed targets, without adding origins twice.
    command.goal_time_steps[2] = 4
    command._refresh_goal([2])
    command._debug_vis_callback(None)
    torch.testing.assert_close(drawn["target_root"][0], command.goal_anchor_pos_w[[2]])
    torch.testing.assert_close(drawn["target_object"][0], command.goal_obj_pos_w[[2]])
    drawn.clear()
    command.cfg.goal_vis_env_ids = []
    command._debug_vis_callback(None)
    assert not drawn


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


@pytest.mark.parametrize("draw, expected", [(0.0, [0, 0, 0]), (0.5, [2, 1, 1]), (0.999999, [4, 2, 1])])
@pytest.mark.parametrize("rsi", [False, True])
def test_random_eval_start_and_goal_bounds(command, monkeypatch, draw, expected, rsi):
    command.cfg.eval_mode = True
    command.cfg.eval_start_mode = "random"
    command.cfg.goal_sampling_mode = "random_future"
    command.cfg.rsi = rsi
    command._eval_env_clip = torch.tensor([1, 0, 2])
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.full((n,), draw, device=device))

    command._resample_command(torch.arange(3))

    assert command.env_clip.tolist() == [1, 0, 2]
    assert command.time_steps.tolist() == expected
    assert (command.goal_time_steps > command.time_steps).all()
    assert (command.goal_time_steps < command.motion.clip_lengths[command.env_clip]).all()
    torch.testing.assert_close(command.joint_pos, command.motion.frames(command.env_clip, command.time_steps).joint_pos)
    assert_goal_matches_selected_frame(command)


@pytest.mark.parametrize("clip_mode", ["same_clip", "cross_clip"])
def test_random_eval_restores_full_reference_state(command, monkeypatch, clip_mode):
    command.cfg.eval_mode = True
    command.cfg.eval_start_mode = "random"
    command.cfg.eval_clip_mode = clip_mode
    command.cfg.goal_sampling_mode = "random_future"
    command._eval_env_clip = torch.tensor([1, 0, 2])
    buffer = command.motion.buffer
    frame = torch.arange(13).float()
    angle = frame / 13
    quat = torch.stack((angle.cos(), angle.sin(), frame * 0, frame * 0), dim=-1)
    buffer.body_quat_w[:] = quat[:, None, :]
    buffer.object_quat_w[:] = quat
    buffer.joint_vel[:] = frame[:, None] + 1
    buffer.body_lin_vel_w[:] = frame[:, None, None] + 2
    buffer.body_ang_vel_w[:] = frame[:, None, None] + 3
    buffer.object_lin_vel_w[:] = frame[:, None] + 4
    buffer.object_ang_vel_w[:] = frame[:, None] + 5
    writes = {}

    def capture(name):
        def write(*values, env_ids):
            writes[name] = tuple(value.clone() for value in values)
            torch.testing.assert_close(env_ids, torch.arange(3))
        return write

    monkeypatch.setattr(command.robot, "write_root_state_to_sim", capture("root"))
    monkeypatch.setattr(command.robot, "write_joint_state_to_sim", capture("joints"))
    monkeypatch.setattr(command.object, "write_root_state_to_sim", capture("object"))
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.full((n,), 0.5, device=device))

    command._resample_command(torch.arange(3))

    reference = command.motion.frames(command.init_clip_ids, command.init_time_steps)
    origins = command._env.scene.env_origins
    torch.testing.assert_close(writes["root"][0], torch.cat((
        reference.body_pos_w[:, 0] + origins, reference.body_quat_w[:, 0],
        reference.body_lin_vel_w[:, 0], reference.body_ang_vel_w[:, 0],
    ), dim=-1))
    torch.testing.assert_close(writes["joints"][0], reference.joint_pos)
    torch.testing.assert_close(writes["joints"][1], reference.joint_vel)
    torch.testing.assert_close(writes["object"][0], torch.cat((
        reference.object_pos_w + origins, reference.object_quat_w,
        reference.object_lin_vel_w, reference.object_ang_vel_w,
    ), dim=-1))
    running_reference = command.motion.frames(command.env_clip, command.time_steps)
    torch.testing.assert_close(command.joint_pos, running_reference.joint_pos)
    torch.testing.assert_close(command.init_time_steps, command.time_steps)
    if clip_mode == "cross_clip":
        assert (command.env_clip != command.init_clip_ids).all()
        assert not torch.equal(reference.joint_pos, running_reference.joint_pos)
    assert_goal_matches_selected_frame(command)


def test_random_eval_partial_reset_preserves_other_envs(command, monkeypatch):
    command.cfg.eval_mode = True
    command.cfg.eval_start_mode = "random"
    command.cfg.goal_sampling_mode = "random_future"
    command._eval_env_clip = torch.tensor([1, 0, 2])
    before = (command.env_clip.clone(), command.time_steps.clone(),
              command.goal_time_steps.clone(), command.goal_joint_pos.clone())
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.full((n,), 0.5, device=device))

    command._resample_command([0])

    assert command.env_clip[0] == 1
    assert command.time_steps[0] == 2
    assert command.goal_time_steps[0] == 4
    after = (command.env_clip, command.time_steps, command.goal_time_steps, command.goal_joint_pos)
    for old, new in zip(before, after):
        torch.testing.assert_close(new[1:], old[1:])


def test_random_eval_single_frame_clip(command):
    command.cfg.eval_mode = True
    command.cfg.eval_start_mode = "random"
    command.cfg.goal_sampling_mode = "random_future"
    command.motion.clip_lengths[0] = 1
    command._eval_env_clip = torch.zeros(3, dtype=torch.long)

    command._resample_command(torch.arange(3))

    assert command.time_steps.tolist() == [0, 0, 0]
    assert command.goal_time_steps.tolist() == [0, 0, 0]
    assert_goal_matches_selected_frame(command)


def test_random_eval_reset_reproducible_with_fixed_seed(command):
    command.cfg.eval_mode = True
    command.cfg.eval_start_mode = "random"
    command.cfg.goal_sampling_mode = "random_future"
    command._eval_env_clip = torch.tensor([1, 0, 2])
    draws = []
    with torch.random.fork_rng(devices=[]):
        for _ in range(2):
            torch.manual_seed(42)
            command._resample_command(torch.arange(3))
            draws.append((command.time_steps.clone(), command.goal_time_steps.clone()))
    torch.testing.assert_close(draws[0], draws[1])


def test_invalid_eval_start_mode_rejected(command):
    cfg = SimpleNamespace(goal_sampling_mode="random_future", eval_start_mode="invalid")
    with pytest.raises(ValueError, match="Invalid eval_start_mode"):
        type(command)(cfg, None)


def test_invalid_eval_clip_mode_rejected(command):
    cfg = SimpleNamespace(goal_sampling_mode="random_future", eval_start_mode="zero", eval_clip_mode="invalid")
    with pytest.raises(ValueError, match="Invalid eval_clip_mode"):
        type(command)(cfg, None)


@pytest.mark.parametrize("draw, expected", [(0.0, [0, 0, 0]), (0.5, [1, 1, 1]), (0.999999, [2, 2, 1])])
def test_cross_clip_start_bounds_leave_a_future_goal(command, monkeypatch, draw, expected):
    command.cfg.eval_mode = True
    command.cfg.eval_clip_mode = "cross_clip"
    command.cfg.eval_start_mode = "random"
    command.cfg.goal_sampling_mode = "random_future"
    command._eval_env_clip = torch.tensor([1, 0, 2])
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.full((n,), draw, device=device))

    command._resample_command(torch.arange(3))

    assert command.init_clip_ids.tolist() == [1, 0, 2]
    assert command.init_time_steps.tolist() == expected
    assert command.time_steps.tolist() == expected
    assert (command.env_clip != command.init_clip_ids).all()
    assert (command.goal_time_steps > command.time_steps).all()
    assert (command.goal_time_steps < command.motion.clip_lengths[command.env_clip]).all()
    assert_goal_matches_selected_frame(command)


def test_cross_clip_sampling_filters_object_and_length(command):
    command.motion.clip_object = torch.tensor([0, 0, 1])
    assert command._sample_cross_clip(torch.tensor([0]), torch.tensor([2])).tolist() == [1]
    # A=1 at t=3: clip 0 has no future frame and clip 2 is another object.
    with pytest.raises(ValueError, match="strictly future"):
        command._sample_cross_clip(torch.tensor([1]), torch.tensor([3]))


def test_cross_clip_sampling_can_choose_each_eligible_clip(command):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        selected = command._sample_cross_clip(torch.zeros(128, dtype=torch.long), torch.zeros(128, dtype=torch.long))
    assert set(selected.tolist()) == {1, 2}


@pytest.mark.parametrize("lengths,objects", [([4, 6, 3], [0, 1, 1]), ([4, 1, 1], [0, 0, 0])])
def test_cross_clip_reset_rejects_missing_compatible_reference(command, lengths, objects):
    command.cfg.eval_mode = True
    command.cfg.eval_clip_mode = "cross_clip"
    command.motion.clip_lengths = torch.tensor(lengths)
    command.motion.clip_object = torch.tensor(objects)
    command._eval_env_clip = torch.tensor([0, 1, 2])
    before = command.env_clip.clone()
    with pytest.raises(ValueError, match="no compatible reference"):
        command._resample_command([0])
    torch.testing.assert_close(command.env_clip, before)


def test_cross_clip_can_initialize_from_a_single_frame_clip(command):
    command.cfg.eval_mode = True
    command.cfg.eval_clip_mode = "cross_clip"
    command.cfg.eval_start_mode = "random"
    command.cfg.goal_sampling_mode = "random_future"
    command.motion.clip_lengths[0] = 1
    command._eval_env_clip = torch.tensor([0, 1, 2])

    command._resample_command([0])

    assert command.init_clip_ids[0] == 0
    assert command.init_time_steps[0] == command.time_steps[0] == 0
    assert command.env_clip[0] != 0
    assert command.goal_time_steps[0] > 0


def test_cross_clip_goals_stay_in_b_without_resetting_state(command, monkeypatch):
    command.cfg.eval_mode = True
    command.cfg.eval_clip_mode = "cross_clip"
    command.cfg.eval_start_mode = "zero"
    command.cfg.goal_sampling_mode = "random_future"
    command.motion.clip_object = torch.tensor([0, 0, 1])
    command._eval_env_clip = torch.tensor([1, 0, 2])
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.zeros(n, device=device))
    command._resample_command([0])
    assert command.env_clip[0] == 0
    assert command.goal_time_steps[0] == 1
    assert command.first_goal_time_steps[0] == 1

    def reject_reset(*args, **kwargs):
        pytest.fail("Advancing B's goals must not change B or reset simulation state")

    monkeypatch.setattr(command, "_sample_cross_clip", reject_reset)
    monkeypatch.setattr(command.robot, "write_root_state_to_sim", reject_reset)
    monkeypatch.setattr(command.robot, "write_joint_state_to_sim", reject_reset)
    monkeypatch.setattr(command.object, "write_root_state_to_sim", reject_reset)
    for expected_goal in (2, 3, 3, 3):
        command._update_command()
        assert command.env_clip[0] == 0
        assert command.init_clip_ids[0] == 1
        assert command.init_time_steps[0] == 0
        assert command.goal_time_steps[0] == expected_goal
        assert command.first_goal_time_steps[0] == 1
        assert_goal_matches_selected_frame(command)
    assert command.time_steps[0] == 3  # Ends at B's last frame, not A's frame 5.


def test_cross_clip_partial_reset_and_seed_reproducibility(command):
    command.cfg.eval_mode = True
    command.cfg.eval_clip_mode = "cross_clip"
    command.cfg.eval_start_mode = "random"
    command.cfg.goal_sampling_mode = "random_future"
    command._eval_env_clip = torch.tensor([1, 0, 2])
    unchanged = tuple(value[1:].clone() for value in (
        command.env_clip, command.time_steps, command.init_clip_ids, command.init_time_steps,
        command.goal_time_steps, command.goal_joint_pos, command.first_goal_time_steps,
    ))
    draws = []
    with torch.random.fork_rng(devices=[]):
        for _ in range(2):
            torch.manual_seed(42)
            command._resample_command([0])
            draws.append(tuple(value.clone() for value in (
                command.env_clip, command.time_steps, command.goal_time_steps,
            )))
    torch.testing.assert_close(draws[0], draws[1])
    for before, after in zip(unchanged, (
        command.env_clip, command.time_steps, command.init_clip_ids, command.init_time_steps,
        command.goal_time_steps, command.goal_joint_pos, command.first_goal_time_steps,
    )):
        torch.testing.assert_close(after[1:], before)


@pytest.fixture
def reference_checks(command, isaac_math):
    command.cfg.eval_mode = True
    command.cfg.eval_clip_mode = "cross_clip"
    command.first_goal_time_steps[:] = torch.tensor([2, 2, 4])
    command.time_steps[:] = torch.tensor([1, 2, 5])  # Before, at, and after the first goal.
    command._refresh_caches()
    command.robot.find_bodies = lambda names, **kwargs: ([0, 1], ["left", "right"])
    command.robot.data.GRAVITY_VEC_W = torch.tensor([0.0, 0, -1]).repeat(3, 1)
    command.robot.data.joint_pos = torch.zeros(3, 3)
    command.robot.data.joint_vel = torch.zeros(3, 3)
    command.robot.data.body_quat_w[:] = torch.tensor([0.0, 1, 0, 0])
    command.object.data = SimpleNamespace(
        root_pos_w=command._env.scene.env_origins.clone(),
        root_quat_w=torch.tensor([0.0, 1, 0, 0]).repeat(3, 1),
    )
    command.metrics = {f"error_{name}": torch.full((3,), 123.0) for name in (
        "anchor_pos", "anchor_rot", "body_pos", "body_rot", "joint_pos", "joint_vel", "object_pos", "object_rot"
    )}
    env = SimpleNamespace(
        num_envs=3, device="cpu", command_manager=SimpleNamespace(get_term=lambda name: command),
        scene={
            "robot": command.robot,
            "contact_sensor": SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(3, 1, 2, 3))),
        },
    )
    ns = definitions(
        PACKAGE / "tasks/hoi/mdp/terminations.py",
        {"bad_anchor_pos", "bad_anchor_ori", "bad_object_pos", "bad_object_ori",
         "bad_motion_body_pos", "bad_motion_body_pos_z_only", "bad_contact", "motion_clip_end"},
        dict(torch=torch, math_utils=isaac_math, ManagerTermBase=object),
    )
    calls = {
        "bad_anchor_pos": {}, "bad_object_pos": {},
        "bad_anchor_ori": {"asset_cfg": SimpleNamespace(name="robot")},
        "bad_object_ori": {"asset_cfg": SimpleNamespace(name="robot")},
        "bad_motion_body_pos": {"body_names": ["left", "right"]},
        "bad_motion_body_pos_z_only": {"body_names": ["left", "right"]},
    }
    return command, env, ns, calls


def test_cross_clip_reference_errors_start_immediately(reference_checks):
    command, env, ns, calls = reference_checks
    command._update_metrics()
    assert (command.metrics["error_anchor_pos"] > 0).all()
    for name, kwargs in calls.items():
        assert ns[name](env, command_name="motion", threshold=0.25, **kwargs).all(), name


def assert_goal_matches_selected_frame(command):
    frames = command.motion.frames(command.env_clip, command.goal_time_steps)
    torch.testing.assert_close(command.goal_joint_pos, frames.joint_pos)
    torch.testing.assert_close(command._goal_anchor_pos, frames.body_pos_w[:, command.anchor_index])
    torch.testing.assert_close(command.goal_anchor_quat_w, frames.body_quat_w[:, command.anchor_index])
    torch.testing.assert_close(command._goal_obj_pos, frames.object_pos_w)
    torch.testing.assert_close(command.goal_obj_quat_w, frames.object_quat_w)


@pytest.mark.parametrize("draw, expected", [(0.0, [1, 1, 2]), (0.5, [2, 2, 4]), (0.999999, [3, 2, 5])])
def test_random_goal_sampling_uses_each_clips_future_interval(command, monkeypatch, draw, expected):
    command.cfg.goal_sampling_mode = "random_future"
    command.time_steps = torch.tensor([0, 0, 1])
    # Distinguish orientation frames as well as position/joint frames.
    angle = torch.arange(13).float() / 13
    quats = torch.stack((angle.cos(), torch.zeros(13), torch.zeros(13), angle.sin()), dim=-1)
    command.motion.buffer.body_quat_w[:] = quats[:, None, :]
    command.motion.buffer.object_quat_w[:] = quats
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.full((n,), draw, device=device))

    command._resample_goal()

    assert command.goal_time_steps.tolist() == expected
    assert_goal_matches_selected_frame(command)


def test_random_goals_hold_and_switch_at_reference_time_without_reset(command, monkeypatch):
    command.cfg.goal_sampling_mode = "random_future"
    command.cfg.eval_mode = True
    command.time_steps = torch.tensor([0, 0, 1])
    command.goal_time_steps = torch.tensor([2, 2, 4])
    command._refresh_goal()
    before = command.goal_joint_pos.clone()

    def reject_sim_write(*args, **kwargs):
        pytest.fail("A goal switch must not reset robot/object state")

    monkeypatch.setattr(command.robot, "write_root_state_to_sim", reject_sim_write)
    monkeypatch.setattr(command.robot, "write_joint_state_to_sim", reject_sim_write)
    monkeypatch.setattr(command.object, "write_root_state_to_sim", reject_sim_write)
    draws = []

    def rand(n, *, device):
        draws.append(n)
        return torch.zeros(n, device=device)

    monkeypatch.setattr(torch, "rand", rand)

    command._update_command()  # t = [1, 1, 2]: no goal reached.
    assert command.goal_time_steps.tolist() == [2, 2, 4]
    torch.testing.assert_close(command.goal_joint_pos, before)
    assert draws == []

    command._update_command()  # t = [2, 2, 3]: only env 0 needs a new goal.
    assert command.goal_time_steps.tolist() == [3, 2, 4]
    torch.testing.assert_close(command.goal_joint_pos[1:], before[1:])
    assert draws == [1]
    assert_goal_matches_selected_frame(command)

    command._update_command()  # t = [3, 2, 4]: only env 2 needs a new goal.
    assert command.goal_time_steps.tolist() == [3, 2, 5]
    assert draws == [1, 1]
    assert_goal_matches_selected_frame(command)

    for _ in range(3):
        command._update_command()
    assert command.time_steps.tolist() == [3, 2, 5]
    assert command.goal_time_steps.tolist() == [3, 2, 5]
    assert draws == [1, 1]  # Final goals are held without resampling.


def test_random_goal_partial_rsi_reset_preserves_other_envs(command, monkeypatch):
    command.cfg.goal_sampling_mode = "random_future"
    before = command.goal_joint_pos.clone()
    before_steps = command.goal_time_steps.clone()
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.full((n,), 0.25, device=device))

    command._resample_command([0])

    assert command.env_clip.tolist() == [1, 2, 1]
    assert command.time_steps[0].item() == 1
    assert command.goal_time_steps[0].item() == 3
    torch.testing.assert_close(command.goal_time_steps[1:], before_steps[1:])
    torch.testing.assert_close(command.goal_joint_pos[1:], before[1:])
    torch.testing.assert_close(command.joint_pos, command.motion.frames(command.env_clip, command.time_steps).joint_pos)
    assert_goal_matches_selected_frame(command)


@pytest.mark.parametrize("new_clip, start, goal", [(0, 0, 1), (1, 1, 3)])
def test_random_goal_rollover_samples_once_for_new_clip(command, monkeypatch, new_clip, start, goal):
    command.cfg.goal_sampling_mode = "random_future"
    command.env_clip[0] = 1
    command.time_steps = torch.tensor([5, 0, 0])
    command.goal_time_steps = command.motion.clip_lengths[command.env_clip] - 1
    command._refresh_goal()
    before = command.goal_joint_pos.clone()
    command.motion.sample_clip = lambda objects: torch.full_like(objects, new_clip)
    draws = []

    def rand(n, *, device):
        draws.append(n)
        return torch.full((n,), 0.25, device=device)

    monkeypatch.setattr(torch, "rand", rand)

    command._update_command()

    assert command.env_clip[0].item() == new_clip
    assert command.time_steps[0].item() == start
    assert command.goal_time_steps[0].item() == goal
    assert draws == [1, 1]  # One RSI draw and one goal draw; no duplicate goal selection.
    torch.testing.assert_close(command.goal_joint_pos[1:], before[1:])
    torch.testing.assert_close(command.joint_pos, command.motion.frames(command.env_clip, command.time_steps).joint_pos)
    assert_goal_matches_selected_frame(command)


@pytest.mark.parametrize("eval_mode", [False, True])
def test_random_goals_handle_single_frame_clips(command, eval_mode):
    command.cfg.goal_sampling_mode = "random_future"
    command.cfg.eval_mode = eval_mode
    command.motion.clip_lengths[0] = 1
    command.motion.sample_clip = lambda objects: torch.zeros_like(objects)
    command.env_clip.zero_()
    command.time_steps.zero_()
    command._resample_goal()

    for _ in range(3):
        command._update_command()
        assert command.time_steps.tolist() == [0, 0, 0]
        assert command.goal_time_steps.tolist() == [0, 0, 0]
        assert_goal_matches_selected_frame(command)


def test_random_goal_sampling_is_reproducible_with_fixed_seed(command):
    command.cfg.goal_sampling_mode = "random_future"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        command._resample_goal()
        before = command.goal_time_steps.clone()
        torch.manual_seed(42)
        command._resample_goal()
    torch.testing.assert_close(command.goal_time_steps, before)


def test_goal_observations_reanchor_world_target(command, isaac_math):
    ns = definitions(
        PACKAGE / "tasks/hoi/mdp/observations.py",
        {"goal_robot_pose", "goal_object_pose"},
        dict(
            torch=torch,
            matrix_from_quat=isaac_math.matrix_from_quat,
            subtract_frame_transforms=isaac_math.subtract_frame_transforms,
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


def test_task_goal_advances_only_on_arrival_and_uses_goal_frame(command):
    command.cfg.eval_mode = True
    command.cfg.eval_success_mode = "goal_success"
    command.cfg.goal_sampling_mode = "random_future"
    command.goal_time_steps[:] = 1
    command.time_steps[:] = command.motion.clip_lengths[command.env_clip] - 1
    command._update_command()
    assert command.goal_time_steps.tolist() == [1, 1, 1]
    command.goal_reached[0] = True
    command._update_command()
    assert 2 <= command.goal_time_steps[0] <= 3
    assert command.goal_time_steps[1:].tolist() == [1, 1]
    assert not command.goal_reached.any()


@pytest.fixture
def task_check(command, isaac_math):
    class TermBase:
        def __init__(self, cfg, env):
            pass

    ns = definitions(
        PACKAGE / "tasks/hoi/mdp/terminations.py", {"goal_success", "robot_fallen"},
        dict(torch=torch, math_utils=isaac_math, ManagerTermBase=TermBase),
    )
    command.cfg.goal_position_threshold = 0.1
    command.cfg.goal_rotation_threshold_deg = 15.0
    command.cfg.goal_hold_seconds = 0.06  # Three control steps.
    command.cfg.fallen_height_threshold = 0.35
    command.cfg.fallen_tilt_threshold_deg = 75.0
    command.goal_time_steps[:] = torch.tensor([1, 2, 4])
    command._refresh_goal()
    command.object.data = SimpleNamespace(
        root_pos_w=command.goal_obj_pos_w.clone(), root_quat_w=command.goal_obj_quat_w.clone()
    )
    env = SimpleNamespace(
        num_envs=3, device="cpu", step_dt=0.02, scene=command._env.scene,
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    term = ns["goal_success"](None, env)
    return command, env, term, ns


def test_goal_hold_break_reset_and_terminal_snapshot(task_check):
    command, env, term, _ = task_check
    assert not term(env, "motion").any()
    command.object.data.root_pos_w[0, 0] += 1
    assert not term(env, "motion").any()
    command.object.data.root_pos_w[0] = command.goal_obj_pos_w[0]
    assert term(env, "motion").tolist() == [False, True, False]
    assert command.goal_reached.tolist() == [False, True, True]
    assert term.hold.tolist() == [1, 0, 0]
    assert term.snapshot["final_object_position_error"][1] == 0
    assert len(term.snapshot_history[1]) == 1
    term.reset(torch.tensor([1]))
    command.object.data.root_pos_w[1] += 100  # Simulate auto-reset physics state.
    assert term.snapshot["final_object_position_error"][1] == 0
    assert len(term.snapshot_history[1]) == 1
    assert term.history[1] == []
    assert term.elapsed.tolist() == [3, 0, 3]
    term.reset(slice(None))
    assert term.elapsed.tolist() == [0, 0, 0]


def test_goal_orientation_requires_full_rotation_but_accepts_quaternion_sign(task_check):
    command, env, term, _ = task_check
    command.object.data.root_quat_w[0] *= -1  # Same rotation.
    command.object.data.root_quat_w[1] = torch.tensor([0.0, 0, 0, 1.0])  # Pure yaw, same gravity.
    for _ in range(3):
        term(env, "motion")
    assert command.goal_reached.tolist() == [True, False, True]


def test_fall_check_is_independent_of_reference(task_check):
    command, env, _, ns = task_check
    command.robot.data.body_pos_w[:, 0] = env.scene.env_origins
    command.robot.data.body_pos_w[:, 0, 2] += torch.tensor([1.0, 0.2, 1.0])
    command.robot.data.body_quat_w[:, 0] = torch.tensor([1.0, 0, 0, 0])
    command.robot.data.body_quat_w[2, 0] = torch.tensor([0.0, 1, 0, 0])
    assert ns["robot_fallen"](env, "motion").tolist() == [False, True, True]
