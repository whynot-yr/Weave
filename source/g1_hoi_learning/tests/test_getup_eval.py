"""Offline checks for rollout diagnostics, including tolerance and target/state separation."""
import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[3]/'scripts/getup/eval_diagnostics.py'
spec = importlib.util.spec_from_file_location('getup_eval_diagnostics', PATH)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_sampled_limit_exceedance_and_tolerance():
    data = dict(time_s=np.array([.005, .010, .015]),
                joint_pos=np.array([[0., 0.], [1.00001, 0.], [1.2, 0.]]),
                joint_vel=np.array([[0., 0.], [2.5, 0.], [-3., 0.]]),
                joint_target=np.array([[0., 1.5], [0., 1.5], [0., 0.]]),
                computed_torque=np.array([[0., 0.], [10., 0.], [11., 0.]]),
                applied_torque=np.array([[0., 0.], [10., 0.], [10., 0.]]))
    limits = dict(position=np.array([[-1., 1.], [-1., 1.]]),
                  soft_position=np.array([[-.9, .9], [-.9, .9]]),
                  velocity=np.array([2., 2.]), effort=np.array([10., 10.]))
    first, second = diagnostics.joint_summary(data, ['driver', 'mimic'], limits, .005)
    assert first['hard_position_sample_count'] == 1
    assert np.isclose(first['hard_position_max_excess'], .2)
    assert first['hard_position_first_time_s'] == .015
    assert first['hard_position_sampled_duration_s'] == .005
    assert first['soft_position_sample_count'] == 2
    assert first['velocity_sample_count'] == 2
    assert first['velocity_max_excess'] == 1.
    assert first['target_position_sample_count'] == 0
    assert second['target_position_sample_count'] == 2
    assert second['hard_position_sample_count'] == 0
    assert second['hard_position_first_time_s'] is None
    assert np.isclose(first['torque_estimate_saturation_fraction'], 2/3)
