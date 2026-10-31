"""Offline torque audit. Solver joint efforts are never promoted to motor torque."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def longest(mask, dt):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths.max() * dt) if len(lengths) else 0.


def summarize(data, meta):
    names = meta['joint_names']
    t = data['time_s']; dt = meta['physics_dt']
    q = data['joint_pos']; target = data['joint_target']
    bounds = data['limit_position']; soft = data['limit_soft_position']
    caps = data['limit_effort']
    requested = data['computed_torque']; applied = data['applied_torque']
    solver = data.get('solver_projected_joint_effort', np.full_like(q, np.nan))
    ptol = meta.get('position_tolerance_rad', 1e-4)
    specs = meta.get('actuator_joint_info', {})
    rows = []; masks = []
    for j, name in enumerate(names):
        info = specs.get(name, {})
        implicit = info.get('is_implicit', True)
        valid_cap = bool(np.isfinite(caps[j]) and caps[j] > 0)
        tol = max(1e-4, 1e-3 * float(caps[j])) if valid_cap else np.nan
        u = abs(applied[:, j]) / caps[j] if valid_cap else np.full(len(t), np.nan)
        valid = np.isfinite(u)
        sat = valid & (abs(applied[:, j]) >= caps[j] - tol)
        req = np.isfinite(requested[:, j]) & (abs(requested[:, j]) > caps[j] + tol)
        hard_ex = np.maximum(np.maximum(bounds[j, 0]-q[:, j], q[:, j]-bounds[j, 1]), 0)
        soft_ex = np.maximum(np.maximum(soft[j, 0]-q[:, j], q[:, j]-soft[j, 1]), 0)
        target_ex = np.maximum(np.maximum(bounds[j, 0]-target[:, j], target[:, j]-bounds[j, 1]), 0)
        hard = hard_ex > ptol
        near = np.minimum(abs(q[:, j]-bounds[j, 0]), abs(q[:, j]-bounds[j, 1])) <= .001
        # Explicit actuator output is a commanded effort, not a measurement.
        source = 'implicit_pd_estimate' if implicit else 'explicit_actuator_command'
        row = dict(joint=name, role='mimic' if name in meta.get('mimic', {}) else 'drive',
                   actuator_type=info.get('class_type', 'unknown'), torque_source=source,
                   validation_status='motor_output_unverified', samples=len(t), duration_s=len(t)*dt,
                   effort_limit_nm=float(caps[j]), saturation_tolerance_nm=tol,
                   actual_peak_torque_nm=None, actual_torque_rms_nm=None,
                   actual_saturation_ratio=None, actual_near_saturation_ratio=None,
                   torque_status='unknown', combined_status='unknown_torque',
                   hard_position_ratio=float(hard.mean()), hard_position_max_excess_rad=float(hard_ex.max()),
                   soft_position_ratio=float((soft_ex > ptol).mean()),
                   target_position_ratio=float((target_ex > ptol).mean()),
                   near_hard_limit_ratio=float(near.mean()),
                   tracking_error_rms_rad=float(np.sqrt(np.mean((target[:, j]-q[:, j])**2))),
                   request_over_limit_ratio=float(req.mean()) if valid_cap else None,
                   output_saturation_ratio=float(sat.mean()) if valid.all() else None,
                   output_near_saturation_ratio=float((u > .95).mean()) if valid.all() else None,
                   output_saturation_duration_s=float(sat.sum()*dt) if valid.all() else None,
                   output_longest_saturation_s=longest(sat, dt) if valid.all() else None,
                   output_hard_overlap_duration_s=float((sat & hard).sum()*dt) if valid.all() else None,
                   near_limit_output_utilization_max=float(u[near].max()) if near.any() and valid[near].all() else None,
                   joint_status='hard_violation' if hard.any() else 'soft_violation' if (soft_ex > ptol).any() else 'near_limit' if near.any() else 'within_limits')
        row['output_joint_classification'] = (
            'unknown_output' if not valid.all() else
            'both_simultaneous' if (sat & hard).any() else
            'both_at_different_times' if sat.any() and hard.any() else
            'torque_only' if sat.any() else 'joint_only' if hard.any() else
            'near_limit_with_margin' if near.any() and np.max(u[near]) <= .8 else
            'neither')
        for label, values in [('requested', requested[:, j]), ('output', applied[:, j]), ('solver_projected', solver[:, j])]:
            finite = np.isfinite(values)
            row[label+'_valid_fraction'] = float(finite.mean())
            # Partial availability must not silently shrink the metric denominator.
            row[label+'_peak_nm'] = float(abs(values).max()) if finite.all() else None
            row[label+'_rms_nm'] = float(np.sqrt(np.mean(values**2))) if finite.all() else None
            row[label+'_peak_time_s'] = float(t[abs(values).argmax()]) if finite.all() else None
        for label, fn in [('max', np.max), ('mean', np.mean), ('p95', lambda x: np.percentile(x, 95))]:
            row['output_utilization_'+label] = float(fn(u)) if valid.all() else None
        rows.append(row); masks.append(sat)
    return rows, np.asarray(masks).T


def write_audit(data, meta, output, plots=True):
    # Materialize compressed NPZ channels once, not once per CSV cell.
    data = {key: data[key] for key in data}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, sat = summarize(data, meta)
    with (output/'torque_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    names = meta['joint_names']; q = data['joint_pos']; caps = data['limit_effort']; t = data['time_s']
    solver = data.get('solver_projected_joint_effort', np.full_like(q, np.nan))
    actuation = data.get('dof_actuation_force', np.full_like(q, np.nan))
    with (output/'torque_timeseries.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'joint', 'torque_source', 'requested_nm', 'output_nm', 'solver_projected_nm',
                    'actual_motor_torque_nm', 'dof_actuation_force_nm', 'effort_limit_nm', 'output_utilization', 'output_saturated', 'position_rad', 'target_rad'])
        for i, time in enumerate(t):
            for j, name in enumerate(names):
                cap = caps[j]
                w.writerow([time, name, rows[j]['torque_source'], data['computed_torque'][i,j],
                            data['applied_torque'][i,j], solver[i,j], '',
                            actuation[i,j], cap,
                            abs(data['applied_torque'][i,j])/cap if np.isfinite(cap) and cap > 0 else '',
                            int(sat[i,j]), q[i,j], data['joint_target'][i,j]])
    audit_meta = dict(physics_dt=meta['physics_dt'], samples=len(t), joint_count=len(names),
                      saturation_abs_tolerance_nm=1e-4, saturation_relative_tolerance=.001,
                      near_saturation_threshold=.95, near_limit_distance_rad=.001,
                      episode_result=meta.get('episode_result'),
                      actuator_joint_info=meta.get('actuator_joint_info', {}),
                      solver_capture_errors=meta.get('solver_capture_errors', {}),
                      notes=['Output means implicit PD estimate or explicit actuator command, as specified per row.',
                             'Projected solver joint effort is not validated as isolated motor output.',
                             'Actual motor metrics and classifications remain unknown.',
                             'output_joint_classification uses estimates/commands and actual HARD position violations only.',
                             'near_limit_with_margin requires maximum output utilization <= 0.8 while near the limit.',
                             'Durations are sampled counts times physics_dt. All samples are retained.',
                             'Incomplete torque channels yield blank aggregate metrics.'])
    (output/'torque_audit_metadata.json').write_text(json.dumps(audit_meta, indent=2))
    if not plots:
        return rows
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def draw_torque(ax, j):
        for key, label in [('computed_torque', 'requested (estimate/command)'),
                           ('applied_torque', 'output (estimate/command)')]:
            ax.plot(t, data[key][:, j], label=label, lw=.7)
        if np.isfinite(solver[:, j]).any():
            ax.plot(t, solver[:, j], label='solver projected', lw=.7)
        for bound in [-caps[j], caps[j]]:
            ax.axhline(bound, color='red', lw=.7)
        ax.set(title=names[j], xlabel='seconds', ylabel='Nm')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)

    for start in range(0, len(names), 9):
        for kind in ('torque', 'torque_utilization'):
            fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
            for offset, ax in enumerate(axes.flat):
                j = start + offset
                if j >= len(names):
                    ax.set_visible(False)
                    continue
                if kind == 'torque':
                    draw_torque(ax, j)
                else:
                    cap = caps[j]
                    if np.isfinite(cap) and cap > 0:
                        ax.plot(t, abs(data['applied_torque'][:, j])/cap, lw=.7)
                    ax.axhline(.95, color='orange', ls='--', lw=.7)
                    ax.axhline(1, color='red', lw=.7)
                    ax.set(title=names[j], xlabel='seconds', ylabel='Output / limit')
                    ax.grid(alpha=.2)
            fig.suptitle('Actuator estimates/commands; solver projection is not isolated motor torque', fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, .97))
            fig.savefig(output/f'{kind}_{start:02d}.png', dpi=120)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, max(6, len(names)*.22)))
    ax.imshow(sat.T, aspect='auto', interpolation='nearest', extent=(t[0]-meta['physics_dt'], t[-1], len(names)-.5, -.5), vmin=0, vmax=1, cmap='Greys')
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set(xlabel='Time (s)', title='Output saturation (estimate/command only); actual motor saturation UNKNOWN')
    fig.tight_layout(); fig.savefig(output/'torque_saturation.png', dpi=140); plt.close(fig)
    # Retire the old joint-position/torque layouts after torque pages are saved.
    old_names = ['joint_limit_torque.png', 'torque_utilization.png']
    old_names += [f'joint_limit_torque_{start:02d}.png' for start in range(0, len(names), 3)]
    for old_name in old_names:
        (output/old_name).unlink(missing_ok=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    meta = json.loads((args.rollout.parent/'metadata.json').read_text())
    capture = args.rollout.parent/'torque_capture_metadata.json'
    if capture.exists(): meta.update(json.loads(capture.read_text()))
    episodes = args.rollout.parent/'episodes.csv'
    if episodes.exists():
        with episodes.open() as f:
            for episode in csv.DictReader(f):
                if episode['file'] == args.rollout.name:
                    meta['episode_result'] = episode
                    break
    with np.load(args.rollout) as data:
        write_audit(data, meta, args.output, not args.no_plots)
