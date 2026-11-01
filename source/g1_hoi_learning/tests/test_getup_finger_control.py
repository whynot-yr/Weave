"""Check fixed finger control without launching Isaac Sim."""
import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np


PATH = Path(__file__).resolve().parents[3] / 'scripts/getup/finger_control.py'
spec = importlib.util.spec_from_file_location('finger_control', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_control_inputs():
    fingers = [f'{s}_{f}_proximal_joint' for s in ('L', 'R') for f in ('index', 'middle', 'ring', 'pinky')]
    fingers += [f'{s}_thumb_proximal_{a}_joint' for s in ('L', 'R') for a in ('yaw', 'pitch')]
    # Deliberately interleave the action order so masking cannot assume a tail slice.
    names = [f'body_{i}' for i in range(29)]
    for i, name in enumerate(fingers):
        names.insert(i * 2, name)
    mimic = {f'{s}_{f}_intermediate_joint': (f'{s}_{f}_proximal_joint', 1., 0.)
             for s in ('L', 'R') for f in ('index', 'middle', 'ring', 'pinky')}
    mimic.update({f'{s}_thumb_{part}_joint': (f'{s}_thumb_proximal_pitch_joint', ratio, 0.)
                  for s in ('L', 'R') for part, ratio in [('intermediate', 1.6), ('distal', 2.4)]})
    term = SimpleNamespace(_joint_names=names, raw_actions=torch.zeros(2, 41),
                           _scale=2., _offset=torch.arange(82).reshape(2, 41).float() / 100,
                           cfg=SimpleNamespace(mimic=mimic, clip=None))
    robot = SimpleNamespace(joint_names=names + list(mimic),
                            data=SimpleNamespace(joint_pos_limits=torch.tensor([-1., 1.]).repeat(2, 53, 1)))
    return term, robot


def test_fixed_targets_preserve_body_and_policy_and_repeat_across_resets():
    term, robot = make_control_inputs()
    control = module.FixedFingerControl(term, robot)
    for _ in range(3):
        policy = torch.randn(2, 41)
        original = policy.clone()
        result = control.apply(policy)
        assert torch.equal(policy, original)
        body = [i for i in range(41) if i not in control.indices]
        assert torch.equal(result[:, body], policy[:, body])
        targets = result * term._scale + term._offset
        assert torch.allclose(targets[:, control.indices], torch.zeros(2, 12))
        for driver, ratio, offset in term.cfg.mimic.values():
            assert torch.allclose(targets[:, term._joint_names.index(driver)] * ratio + offset, torch.zeros(2))
        term.raw_actions.zero_()  # Action term reset does not disable the override.


@pytest.mark.parametrize('invalid', ['scale', 'wrapper_clip', 'target_clip', 'mimic_limit', 'missing_finger'])
def test_reject_unreachable_or_invalid_fixed_targets(invalid):
    term, robot = make_control_inputs()
    clip = None
    if invalid == 'scale':
        term._scale = 0.
    elif invalid == 'wrapper_clip':
        clip = .001
    elif invalid == 'target_clip':
        term.cfg.clip = {}
        term._clip = torch.tensor([.1, 1.]).repeat(2, 41, 1)
    elif invalid == 'mimic_limit':
        robot.data.joint_pos_limits[:, -1, 0] = .1
    else:
        term._joint_names[0] = 'unexpected'
    with pytest.raises(ValueError):
        module.FixedFingerControl(term, robot, clip)


def test_recorder_captures_executed_actions_before_reset_and_skips_finished_envs():
    path = PATH.with_name('rollout_recorder.py')
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    namespace = {'RecorderTerm': object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    cls = namespace['JointRolloutRecorder']
    recorder = cls.__new__(cls)
    recorder.enabled = True
    recorder.quotas, recorder.completed = [2, 1], [0, 0]
    recorder.control, recorder.action_records = [[], []], [[], []]
    executed = torch.ones(2, 41)
    recorder._env = SimpleNamespace(
        action_manager=SimpleNamespace(action=executed),
        getup_state=SimpleNamespace(success_steps=torch.zeros(2), success=torch.zeros(2, dtype=torch.bool)),
        reset_buf=torch.tensor([False, True]),
        termination_manager=SimpleNamespace(active_terms=['time_out'], get_term=lambda _: torch.tensor([False, True])))
    saved = []

    def save(i, *args):
        saved.append(recorder.action_records[i])
        recorder.action_records[i] = []
        recorder.control[i] = []
        recorder.completed[i] += 1

    recorder._save = save
    policy = torch.full((2, 41), 3.)
    recorder.record_policy_actions(policy)
    policy.zero_()  # The recorder must own a copy of the original policy output.
    recorder.record_post_step()
    executed.zero_()  # Simulate automatic action reset after the terminal snapshot.
    np.testing.assert_array_equal(saved[0][0][0], np.full(41, 3.))
    np.testing.assert_array_equal(saved[0][0][1], np.ones(41))
    recorder.record_policy_actions(policy)
    recorder.record_post_step()
    assert len(recorder.action_records[0]) == 2
    assert recorder.action_records[1] == []  # Finished environment is not recorded again.
