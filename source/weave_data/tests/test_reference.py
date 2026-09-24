import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from weave_data.dp_codec import encode_reference, encode_state, ppo_reference_groups, reanchor_reference
from weave_data.reference_window import local_pose_6d, matrix_from_6d, matrix_from_quat, rotation_6d, window_indices


def test_indices_clamp_and_mask():
    indices, pad = window_indices(
        torch.tensor([2, 0]), torch.tensor([0, 1, 5]), torch.tensor([3, 9]), torch.tensor([0, 3])
    )
    assert indices.tolist() == [[2, 2, 2], [3, 4, 8]]
    assert pad.tolist() == [[False, True, True], [False, False, False]]


def test_state_uses_named_active_joints_and_initial_heading():
    q = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    positions = torch.arange(53).float()
    indices = list(range(39)) + [43, 48]
    state = encode_state(q, positions, positions + 100, q, indices)
    assert state.shape == (88,)
    torch.testing.assert_close(state[:6], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state[6:47], positions[indices])
    torch.testing.assert_close(state[47:], positions[indices] + 100)


def test_rot6d_roundtrip_and_degenerate():
    q = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    torch.testing.assert_close(matrix_from_6d(rotation_6d(q)), matrix_from_quat(q), atol=1e-6, rtol=1e-6)
    m = matrix_from_6d(torch.zeros(3, 6))
    torch.testing.assert_close(m.transpose(-1, -2) @ m, torch.eye(3).expand(3, -1, -1))
    torch.testing.assert_close(torch.linalg.det(m), torch.ones(3))


def test_original_weave_pose_observations_parity():
    """Execute only pure functions from the original files, without importing Sim."""
    weave = Path(__file__).resolve().parents[3]
    math_file = weave.parent / "IsaacLab/source/isaaclab/isaaclab/utils/math.py"
    if not math_file.exists():
        pytest.skip("Local IsaacLab source not installed")
    names = {"quat_conjugate", "quat_inv", "quat_mul", "quat_apply", "matrix_from_quat", "subtract_frame_transforms"}
    nodes = [n for n in ast.parse(math_file.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = {"torch": torch, "ManagerBasedEnv": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(math_file), "exec"), namespace)
    obs_file = weave / "source/g1_hoi_learning/g1_hoi_learning/tasks/hoi/mdp/observations.py"
    names = {
        "motion_future_anchor_pos_b",
        "motion_future_anchor_ori_b",
        "motion_future_obj_pos_b",
        "motion_future_obj_ori_b",
    }
    nodes = [n for n in ast.parse(obs_file.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(obs_file), "exec"), namespace)
    q = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(4, 5, 4), dim=-1)
    p, tp = torch.randn(4, 3), torch.randn(4, 5, 3)
    command = SimpleNamespace(
        cfg=SimpleNamespace(future_offsets=list(range(5))),
        robot_anchor_pos_w=p,
        robot_anchor_quat_w=q,
        future_anchor_pos_w=tp,
        future_anchor_quat_w=target,
        future_obj_pos_w=tp,
        future_obj_quat_w=target,
    )
    env = SimpleNamespace(num_envs=4, command_manager=SimpleNamespace(get_term=lambda _: command))
    actual_pos, actual_rot = local_pose_6d(p[:, None], q[:, None], tp, target)
    for name in names:
        expected = namespace[name](env, "motion")
        actual = actual_pos if "_pos_" in name else actual_rot
        torch.testing.assert_close(actual.reshape(4, -1), expected, atol=3e-6, rtol=3e-6)


def test_encode_and_reanchor_reference():
    batch, horizon = 2, 40

    def quat(*shape):
        return torch.nn.functional.normalize(torch.randn(*shape, 4), dim=-1)

    reference = {
        "joint_pos": torch.randn(batch, horizon, 53),
        "pelvis_pos": torch.randn(batch, horizon, 3),
        "pelvis_quat": quat(batch, horizon),
        "object_pos": torch.randn(batch, horizon, 3),
        "object_quat": quat(batch, horizon),
        "contact": torch.randint(-1, 2, (batch, horizon, 54)).float(),
    }
    p, q, p2, q2 = torch.randn(batch, 3), quat(batch), torch.randn(batch, 3), quat(batch)
    encoded = encode_reference(reference, p, q)
    reframed = reanchor_reference(encoded, p, q, p2, q2)
    torch.testing.assert_close(reframed, encode_reference(reference, p2, q2), atol=4e-6, rtol=4e-6)
    groups = ppo_reference_groups(reframed)
    assert groups["ref_motion_body"].shape == (batch, 310)
    assert groups["ref_motion_object"].shape == (batch, 315)
    torch.testing.assert_close(
        groups["ref_motion_object"][..., -270:], reference["contact"][:, [0, 5, 10, 15, 20]].flatten(-2)
    )
