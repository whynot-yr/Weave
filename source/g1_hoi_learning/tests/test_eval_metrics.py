"""Exercise first-episode eval statistics without starting Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def make_evaluator():
    path = Path(__file__).resolve().parents[3] / "scripts/rsl_rl/eval.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ClipEvaluator"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)

    def make(lengths, starts, init_clips=None, reference_clips=None, task=False):
        n = len(lengths)
        command = SimpleNamespace(
            env_clip=torch.tensor(reference_clips) if reference_clips is not None else torch.arange(n),
            time_steps=torch.tensor(starts), init_time_steps=torch.tensor(starts),
            first_goal_time_steps=torch.tensor(starts) + 1,
            cfg=SimpleNamespace(eval_success_mode="goal_success" if task else "reference"),
            init_clip_ids=torch.tensor(init_clips) if init_clips is not None else torch.arange(n),
            motion=SimpleNamespace(clip_lengths=torch.tensor(lengths)),
            metrics={"error": torch.arange(1, n + 1).float()},
            _update_metrics=lambda: None,
        )
        flags = {name: torch.zeros(n, dtype=torch.bool) for name in ("clip_end", "time_out", "failure")}
        if task:
            flags["goal_success"] = flags.pop("clip_end")
        goal_term = SimpleNamespace(
            snapshot={"final_goal_reached": torch.zeros(n, dtype=torch.bool),
                      "final_object_position_error": torch.arange(n).float()},
            snapshot_history=[[] for _ in range(n)],
        )
        term_mgr = SimpleNamespace(
            active_terms=list(flags), get_term=lambda name: flags[name],
            get_term_cfg=lambda name: SimpleNamespace(time_out=name != "failure", func=goal_term),
        )
        raw = SimpleNamespace(
            device="cpu", num_envs=n, max_episode_length=1000,
            command_manager=SimpleNamespace(get_term=lambda name: command),
            termination_manager=term_mgr,
        )
        evaluator = namespace["ClipEvaluator"](SimpleNamespace(unwrapped=raw), [f"clip{i}" for i in range(n)])
        return evaluator, command, flags

    return make


@pytest.mark.parametrize("length,start", [(5, 0), (5, 2), (5, 3), (1, 0)])
def test_progress_respects_termination_before_command_advance(make_evaluator, length, start):
    evaluator, command, flags = make_evaluator([length], [start])
    seen = []
    # Match ManagerBasedRLEnv: collect, physics, termination, reset, command advance.
    for _ in range(evaluator.max_steps):
        seen.append(int(command.time_steps[0]))
        evaluator.collect()
        flags["clip_end"][:] = command.time_steps >= length - 1
        dones = flags["clip_end"].clone()
        if dones.any():
            command.time_steps[dones] = 0  # Auto-reset must not change saved first-episode metadata.
        command.time_steps.add_(1).clamp_(max=length - 1)
        evaluator.update(dones, flags["clip_end"])
        if evaluator.done:
            break

    assert seen == list(range(start, length))
    result = evaluator.results()
    clip = result["clips"][0]
    assert evaluator.done
    assert clip["start_frame"] == start
    assert clip["length"] == length
    assert clip["remaining_length"] == length - start
    assert clip["steps"] == length - start
    assert clip["metric_steps"] == length - start
    assert clip["success"]
    assert clip["cause"] == "clip_end"
    assert result["progress_rate"] == 1.0
    assert clip["error"] == 1.0


def test_partial_failure_and_finished_envs_stop_accumulating(make_evaluator):
    evaluator, command, flags = make_evaluator([8, 5], [4, 3])
    assert evaluator.max_steps == 9
    evaluator.collect()
    flags["failure"][0] = True
    evaluator.update(torch.tensor([True, False]), torch.tensor([False, False]))
    assert not evaluator.done
    command.metrics["error"][:] = torch.tensor([100.0, 4.0])
    evaluator.collect()
    flags["clip_end"][1] = True
    evaluator.update(torch.tensor([False, True]), torch.tensor([False, True]))
    # A later episode cannot overwrite the first episode or contribute metrics.
    command.metrics["error"].fill_(1000)
    evaluator.collect()
    flags["clip_end"][:] = True
    flags["failure"].zero_()
    evaluator.update(torch.tensor([True, True]), torch.tensor([True, True]))

    result = evaluator.results()
    assert evaluator.done
    assert result["success_rate"] == 0.5
    assert result["progress_rate"] == pytest.approx((0.25 + 1.0) / 2)
    assert result["failures"] == {"failure": 1}
    assert result["metrics_all"]["error"] == pytest.approx(7 / 3)
    assert result["metrics_success"]["error"] == 3.0
    assert result["clips"][0]["steps"] == 1
    assert result["clips"][0]["cause"] == "failure"
    assert result["clips"][1]["steps"] == 2


@pytest.mark.parametrize("end_term", ["clip_end", "time_out"])
def test_failure_takes_precedence_over_simultaneous_timeout(make_evaluator, end_term):
    evaluator, _, flags = make_evaluator([5], [2])
    evaluator.collect()
    flags[end_term][:] = True
    flags["failure"][:] = True
    evaluator.update(torch.tensor([True]), torch.tensor([True]))
    clip = evaluator.results()["clips"][0]
    assert not clip["success"]
    assert clip["cause"] == "failure"


def test_timeout_before_clip_end_is_not_success(make_evaluator):
    evaluator, _, flags = make_evaluator([5], [2])
    evaluator.collect()
    flags["time_out"][:] = True
    evaluator.update(torch.tensor([True]), torch.tensor([True]))
    clip = evaluator.results()["clips"][0]
    assert not clip["success"]
    assert clip["cause"] == "time_out"
    assert clip["progress"] == pytest.approx(1 / 3)


def test_cross_clip_metadata_and_progress_use_b_length_and_survive_reset(make_evaluator):
    evaluator, command, flags = make_evaluator([8, 4], [2, 0], init_clips=[0, 1], reference_clips=[1, 0])
    assert evaluator.max_steps == 13  # B lengths [4, 8], remaining lengths [2, 8].
    for _ in range(2):
        evaluator.collect()
    flags["clip_end"][0] = True
    evaluator.update(torch.tensor([True, False]), torch.tensor([True, False]))
    command.init_clip_ids[:] = torch.tensor([1, 0])
    command.init_time_steps.fill_(1)
    command.env_clip[:] = torch.tensor([0, 1])
    command.time_steps.fill_(1)

    first, second = evaluator.results()["clips"]
    assert first["init_clip_id"] == 0
    assert first["init_clip_name"] == "clip0"
    assert first["init_frame"] == 2
    assert first["reference_clip_id"] == first["clip_id"] == 1
    assert first["reference_clip_name"] == first["name"] == "clip1"
    assert first["reference_start_frame"] == 2
    assert first["reference_length"] == first["length"] == 4
    assert first["remaining_length"] == 2
    assert first["progress"] == 1.0
    assert first["success"]
    assert second["init_clip_id"] == 1
    assert second["reference_clip_id"] == 0
    assert second["progress"] == 0.25


def test_all_steps_record_reference_errors(make_evaluator):
    evaluator, command, flags = make_evaluator([8], [0])
    evaluator.collect()
    assert evaluator.results()["clips"][0]["metric_steps"] == 1
    assert evaluator.results()["metrics_all"]["error"] == 1


def test_task_evaluator_terminal_snapshot_and_failure_priority(make_evaluator):
    evaluator, command, flags = make_evaluator([5, 5, 5], [0, 0, 0], task=True)
    assert evaluator.max_steps == 1001
    evaluator.collect()
    flags["goal_success"][:2] = True
    flags["failure"][1] = True
    flags["time_out"][2] = True
    evaluator.goal_term.snapshot["final_goal_reached"][:2] = True
    evaluator.goal_term.snapshot_history[0] = [{"frame": 4, "elapsed_seconds": 0.5}]
    evaluator.update(torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    result = evaluator.results()
    assert [r["success"] for r in result["clips"]] == [True, False, False]
    assert [r["cause"] for r in result["clips"]] == ["goal_success", "failure", "time_out"]
    assert "progress_rate" not in result
    assert all("progress" not in row for row in result["clips"])
    assert result["failures"] == {"failure": 1, "time_out": 1}
    assert result["clips"][0]["goals_reached"] == 1
    evaluator.goal_term.snapshot["final_object_position_error"].fill_(999)
    evaluator.update(torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    assert evaluator.results()["clips"][0]["final_object_position_error"] == 0
