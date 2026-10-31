"""Offline summaries for Getup rollout diagnostics (no simulator imports)."""
import csv
from pathlib import Path

import numpy as np


def joint_summary(data, names, limits, dt, position_tolerance=1e-4, velocity_tolerance=1e-3):
    q, v, target = data['joint_pos'], data['joint_vel'], data['joint_target']
    for key in ('joint_pos', 'joint_vel', 'joint_target', 'computed_torque', 'applied_torque'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Non-finite {key}; inspect saved rollout before interpreting limit statistics')
    def excess(x, bounds):
        return np.maximum(np.maximum(bounds[:, 0] - x, x - bounds[:, 1]), 0)
    checks = {
        'hard_position': (excess(q, limits['position']), position_tolerance),
        'soft_position': (excess(q, limits['soft_position']), position_tolerance),
        'target_position': (excess(target, limits['position']), position_tolerance),
        'velocity': (np.maximum(np.abs(v) - limits['velocity'], 0), velocity_tolerance),
    }
    rows = []
    for j, name in enumerate(names):
        row = dict(joint=name, position_min_rad=float(q[:, j].min()),
                   position_max_rad=float(q[:, j].max()), max_abs_velocity_rad_s=float(np.abs(v[:, j]).max()),
                   max_abs_tracking_error_rad=float(np.abs(target[:, j]-q[:, j]).max()),
                   max_abs_computed_torque_estimate_nm=float(np.abs(data['computed_torque'][:, j]).max()),
                   max_abs_applied_torque_estimate_nm=float(np.abs(data['applied_torque'][:, j]).max()))
        for key, (values, tolerance) in checks.items():
            mask = values[:, j] > tolerance
            row[key+'_max_excess'] = float(values[:, j].max())
            row[key+'_sample_count'] = int(mask.sum())
            row[key+'_sampled_duration_s'] = float(mask.sum()*dt)
            row[key+'_first_time_s'] = float(data['time_s'][np.flatnonzero(mask)[0]]) if mask.any() else None
        cap = limits['effort'][j]
        saturation = np.abs(data['computed_torque'][:, j]) >= cap if np.isfinite(cap) and cap > 0 else np.zeros(len(q), bool)
        row['torque_estimate_saturation_fraction'] = float(saturation.mean())
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_episode(path, data, names, limits, success_height):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    t = data['time_s']
    # Paginate every joint, including the 12 mimic joints.
    for start in range(0, len(names), 9):
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
        for j, ax in zip(range(start, min(start+9, len(names))), axes.flat):
            ax.plot(t, data['joint_pos'][:, j], label='actual')
            ax.plot(t, data['joint_target'][:, j], label='target', alpha=.65)
            for bound in limits['position'][j]: ax.axhline(bound, color='r', linewidth=.6)
            for bound in limits['soft_position'][j]: ax.axhline(bound, color='orange', linestyle='--', linewidth=.6)
            ax.set_title(names[j], fontsize=8)
            ax.set(xlabel='seconds', ylabel='rad')
        for ax in list(axes.flat)[min(9, len(names)-start):]: ax.set_visible(False)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path/f'position_{start:02d}.png')
        plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
        for j, ax in zip(range(start, min(start+9, len(names))), axes.flat):
            ax.plot(t, data['joint_vel'][:, j])
            ax.axhline(limits['velocity'][j], color='r', linewidth=.6)
            ax.axhline(-limits['velocity'][j], color='r', linewidth=.6)
            ax.set_title(names[j], fontsize=8)
            ax.set(xlabel='seconds', ylabel='rad/s')
        for ax in list(axes.flat)[min(9, len(names)-start):]: ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(path/f'velocity_{start:02d}.png')
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(t, data['pelvis_height'], label='pelvis')
    ax.axhline(success_height, color='r', label='success threshold')
    ax.set(xlabel='seconds', ylabel='height (m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path/'pelvis_height.png')
    plt.close(fig)
