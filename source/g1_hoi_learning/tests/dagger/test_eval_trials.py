"""Repeatable initial conditions, batch assignment, pose geometry and trial reports."""

import math
from types import SimpleNamespace

import pytest
import torch
from g1_hoi_learning.algorithms.dagger.evaluation import assign_eval_trials, build_eval_trials, summarize
from g1_hoi_learning.algorithms.perturbations import offset_object_pose


def test_trial_conditions_are_independent_of_clip_selection_order_and_trial_count():
    names = ["clip0", "clip1", "clip2"]
    all_trials = build_eval_trials(names, None, 3, 42, "initial_small")
    selected = build_eval_trials(names, [2, 0], 5, 42, "initial_small")
    original = {(t["clip_id"], t["trial_id"]): t for t in all_trials}
    for trial in selected:
        if trial["trial_id"] < 3:
            assert trial == original[trial["clip_id"], trial["trial_id"]]
        offset = trial["object_pose_offset"]
        assert abs(offset["x_m"]) <= 0.002 and abs(offset["y_m"]) <= 0.002
        assert abs(offset["yaw_rad"]) <= math.radians(0.5)
    assert len({tuple(t["object_pose_offset"].values()) for t in all_trials}) == len(all_trials)
    changed = build_eval_trials(names, None, 3, 43, "initial_small")
    assert changed[0]["object_pose_offset"] != all_trials[0]["object_pose_offset"]


def test_unperturbed_trials_keep_zero_offsets_and_default_clip_order():
    trials = build_eval_trials(["clip0", "clip1"], None, 1, 42, "none")
    assert [trial["clip_id"] for trial in trials] == [0, 1]
    assert all(all(value == 0 for value in trial["object_pose_offset"].values()) for trial in trials)


@pytest.mark.parametrize(
    "clip_ids,num_trials,seed,profile",
    [
        ([], 1, 42, "none"),
        ([0, 0], 1, 42, "none"),
        ([-1], 1, 42, "none"),
        ([2], 1, 42, "none"),
        (None, 0, 42, "none"),
        (None, 1, -1, "none"),
        (None, 1, 42, "unknown"),
    ],
)
def test_invalid_trials_are_rejected(clip_ids, num_trials, seed, profile):
    with pytest.raises(ValueError):
        build_eval_trials(["clip0", "clip1"], clip_ids, num_trials, seed, profile)


def test_parallel_repeats_and_last_batch_keep_trial_offsets():
    command = SimpleNamespace(
        cfg=SimpleNamespace(eval_mode=True, eval_clip_ids=[]),
        motion=SimpleNamespace(num_objects=1, num_clips=2),
        num_envs=4,
        _eval_env_clip=torch.zeros(4, dtype=torch.long),
        _eval_object_pose_offsets=torch.zeros(4, 3),
    )
    trials = build_eval_trials(["clip0", "clip1"], [1], 5, 42, "initial_small")
    assign_eval_trials(command, trials[:4])
    assert command.cfg.eval_clip_ids == [1, 1, 1, 1]
    expected = torch.tensor([list(trial["object_pose_offset"].values()) for trial in trials])
    torch.testing.assert_close(command._eval_object_pose_offsets, expected[:4])
    assign_eval_trials(command, trials[4:])
    torch.testing.assert_close(command._eval_object_pose_offsets, expected[4:].expand(4, -1))
    # Reusing the scene for an unperturbed run must clear previously assigned offsets.
    assign_eval_trials(command, build_eval_trials(["clip0", "clip1"], [0], 1, 42, "none"))
    assert not command._eval_object_pose_offsets.any()


def test_pose_offset_uses_world_yaw_and_preserves_reference_height_and_velocities():
    # A 90-degree roll followed by WORLD yaw 90 degrees gives wxyz=(.5,.5,.5,.5).
    state = torch.tensor([[1.0, 2.0, 3.0, math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    original = state.clone()
    shifted = offset_object_pose(state, torch.tensor([[0.002, -0.001, math.pi / 2]]))
    torch.testing.assert_close(shifted[:, :3], torch.tensor([[1.002, 1.999, 3.0]]))
    torch.testing.assert_close(shifted[:, 3:7], torch.full((1, 4), 0.5))
    assert torch.equal(shifted[:, 7:], state[:, 7:])
    assert torch.equal(state, original)
    assert torch.equal(offset_object_pose(state, torch.zeros(1, 3)), original)


def test_repeated_trial_statistics_group_by_clip_and_include_uncertainty():
    rows = [
        dict(
            clip_id=clip_id,
            trial_id=trial_id,
            name=f"clip{clip_id}",
            steps=10,
            progress=1.0 if success else 0.5,
            finished=True,
            success=success,
            cause="clip_end" if success else "object_pos",
            tracking=0.1,
        )
        for clip_id, trial_id, success in [(0, 0, True), (0, 1, False), (0, 2, True), (1, 0, False), (1, 1, False)]
    ]
    result = summarize(rows, ["tracking"])
    assert result["num_clips"] == 2 and result["num_trials"] == 5
    assert result["success_rate"] == 0.4 and result["num_successes"] == 2
    first, second = result["clip_summary"]
    assert first["num_trials"] == 3 and first["success_rate"] == pytest.approx(2 / 3)
    assert second["num_trials"] == 2 and second["success_rate"] == 0
    assert second["failures"] == {"object_pos": 2}
    assert second["success_rate_ci95"][1] > 0
    lower, upper = result["success_rate_ci95"]
    assert 0 < lower < 0.4 < upper < 1
