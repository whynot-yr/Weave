"""Metric and missing-source tests; no simulator required."""
import importlib.util
from pathlib import Path
import numpy as np

PATH = Path(__file__).resolve().parents[3]/'scripts/getup/torque_audit.py'
spec = importlib.util.spec_from_file_location('torque_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    data = dict(time_s=np.arange(1, 5)*.005,
                joint_pos=np.array([[0.], [1.1], [0.], [1.]]),
                joint_target=np.ones((4,1))*2,
                computed_torque=np.array([[0.], [12.], [-12.], [9.6]]),
                applied_torque=np.array([[0.], [10.], [-10.], [9.6]]),
                limit_position=np.array([[-1., 1.]]),
                limit_soft_position=np.array([[-.9, .9]]), limit_effort=np.array([10.]))
    return data, dict(joint_names=['joint'], physics_dt=.005)


def test_estimate_metrics_do_not_claim_actual_saturation():
    data, meta = fixture()
    row = audit.summarize(data, meta)[0][0]
    assert row['output_saturation_ratio'] == .5
    assert row['output_near_saturation_ratio'] == .75
    assert row['output_longest_saturation_s'] == .01
    assert row['output_hard_overlap_duration_s'] == .005
    assert np.isclose(row['output_rms_nm'], np.sqrt((100+100+9.6**2)/4))
    assert row['output_joint_classification'] == 'both_simultaneous'
    assert row['actual_saturation_ratio'] is None
    assert row['torque_status'] == 'unknown'
    assert row['solver_projected_peak_nm'] is None


def test_solver_overload_is_not_promoted_to_motor_overload():
    data, meta = fixture()
    data['solver_projected_joint_effort'] = np.full((4,1), 100.)
    row = audit.summarize(data, meta)[0][0]
    assert row['solver_projected_peak_nm'] == 100
    assert row['actual_peak_torque_nm'] is None
    data['solver_projected_joint_effort'][1,0] = np.nan
    row = audit.summarize(data, meta)[0][0]
    assert row['solver_projected_valid_fraction'] == .75
    assert row['solver_projected_rms_nm'] is None


def test_missing_limit_and_nonoverlapping_events():
    data, meta = fixture()
    data['joint_pos'][:] = 0
    data['joint_pos'][0,0] = 1.1
    row = audit.summarize(data, meta)[0][0]
    assert row['output_joint_classification'] == 'both_at_different_times'
    assert row['output_hard_overlap_duration_s'] == 0
    data['limit_effort'][0] = 0
    row = audit.summarize(data, meta)[0][0]
    assert row['output_saturation_ratio'] is None
    assert row['output_utilization_max'] is None


def test_export_retains_every_sample(tmp_path):
    import csv, json
    data, meta = fixture()
    audit.write_audit(data, meta, tmp_path, plots=False)
    rows = list(csv.DictReader((tmp_path/'torque_timeseries.csv').open()))
    assert len(rows) == 4
    assert all(r['actual_motor_torque_nm'] == '' for r in rows)
    assert json.loads((tmp_path/'torque_audit_metadata.json').read_text())['samples'] == 4
