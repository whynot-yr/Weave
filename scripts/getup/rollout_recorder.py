"""Evaluation-only recorder: fresh PhysX samples and pre-reset episode results."""
import json
from pathlib import Path

import numpy as np
from isaaclab.managers.recorder_manager import RecorderTerm

from eval_diagnostics import joint_summary, plot_episode, write_csv


class JointRolloutRecorder(RecorderTerm):
    enabled = False

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.getup_eval_recorder = self

    def start(self, output, episodes, position_tolerance, velocity_tolerance, plots):
        self.output = Path(output)
        self.robot = self._env.scene['robot']
        self.names = list(self.robot.joint_names)
        self.quotas = [episodes//self._env.num_envs + (i < episodes % self._env.num_envs)
                       for i in range(self._env.num_envs)]
        self.completed = [0]*self._env.num_envs
        self.physics = [[] for _ in self.quotas]
        self.control = [[] for _ in self.quotas]
        self.results, self.summary = [], []
        self.position_tolerance, self.velocity_tolerance = position_tolerance, velocity_tolerance
        self.plots = plots
        self.dt = self._env.physics_dt
        d = self.robot.data
        self.limits = {k: v.detach().cpu().numpy().copy() for k, v in {
            'position': d.joint_pos_limits, 'soft_position': d.soft_joint_pos_limits,
            'velocity': d.joint_vel_limits, 'effort': d.joint_effort_limits}.items()}
        self.initial_q = d.joint_pos.detach().cpu().numpy().copy()
        self.initial_v = d.joint_vel.detach().cpu().numpy().copy()
        self.initial_root = d.root_state_w.detach().cpu().numpy().copy()
        self.enabled = True

    @property
    def done(self):
        return self.completed == self.quotas

    def record_post_physics_decimation_step(self):
        if not self.enabled or self.done:
            return None, None
        # This callback runs BEFORE scene.update(). ArticulationData caches can still
        # contain the PRE-step state: read fresh state directly from the PhysX view.
        view, d = self.robot.root_physx_view, self.robot.data
        arrays = {k: v.detach().cpu().numpy().copy() for k, v in {
            'joint_pos': view.get_dof_positions(), 'joint_vel': view.get_dof_velocities(),
            'joint_target': d.joint_pos_target,
            # Implicit actuator estimates computed when applying THIS substep's target.
            'computed_torque': d.computed_torque, 'applied_torque': d.applied_torque,
            'pelvis_height': view.get_root_transforms()[:, 2]-self._env.scene.env_origins[:, 2],
        }.items()}
        for i, quota in enumerate(self.quotas):
            if self.completed[i] < quota:
                self.physics[i].append({k: v[i].copy() for k, v in arrays.items()})
        return None, None

    def record_post_step(self):
        if not self.enabled or self.done:
            return None, None
        # ManagerBasedRLEnv calls this after reward/success computation, before auto-reset.
        s = self._env.getup_state
        for i, quota in enumerate(self.quotas):
            if self.completed[i] >= quota:
                continue
            self.control[i].append((int(s.success_steps[i]), bool(s.success[i])))
            if self._env.reset_buf[i]:
                causes = [name for name in self._env.termination_manager.active_terms
                          if self._env.termination_manager.get_term(name)[i]]
                self._save(i, True, bool(s.success[i]), int(s.success_steps[i]), causes)
        return None, None

    def record_post_reset(self, env_ids):
        if self.enabled:
            ids = range(self._env.num_envs) if env_ids is None else env_ids.tolist()
            for i in ids:
                self.initial_q[i] = self.robot.data.joint_pos[i].detach().cpu().numpy()
                self.initial_v[i] = self.robot.data.joint_vel[i].detach().cpu().numpy()
                self.initial_root[i] = self.robot.data.root_state_w[i].detach().cpu().numpy()
        return None, None

    def _save(self, i, complete, success, hold_steps, causes):
        if not self.physics[i]:
            return
        episode = self.completed[i]
        stem = f'rollout_env{i:03d}_episode{episode:04d}'
        data = {key: np.stack([row[key] for row in self.physics[i]]) for key in self.physics[i][0]}
        data['time_s'] = (np.arange(len(self.physics[i]))+1)*self.dt
        controls = np.asarray(self.control[i]).reshape(-1, 2)
        data.update(control_time_s=(np.arange(len(controls))+1)*self._env.step_dt,
                    success_steps=controls[:, 0].astype(int), success=controls[:, 1].astype(bool),
                    initial_joint_pos=self.initial_q[i].copy(), initial_joint_vel=self.initial_v[i].copy(),
                    initial_root_state_w=self.initial_root[i].copy())
        limits = {k: v[i] for k, v in self.limits.items()}
        np.savez_compressed(self.output/f'{stem}.npz', **data,
                            **{f'limit_{k}': v for k, v in limits.items()})
        rows = joint_summary(data, self.names, limits, self.dt, self.position_tolerance, self.velocity_tolerance)
        self.summary.extend([dict(env_id=i, episode_id=episode, **row) for row in rows])
        self.results.append(dict(env_id=i, episode_id=episode, complete=complete,
                                 success=success, final_hold=hold_steps >= self._env.getup_state.required_steps,
                                 final_hold_steps=hold_steps, max_hold_steps=int(controls[:, 0].max()) if len(controls) else 0,
                                 physics_samples=len(data['time_s']), control_steps=len(controls),
                                 duration_s=float(data['time_s'][-1]), end_causes=';'.join(causes),
                                 max_pelvis_height_m=float(data['pelvis_height'].max()),
                                 final_pelvis_height_m=float(data['pelvis_height'][-1]), file=stem+'.npz'))
        write_csv(self.output/'joint_summary.csv', self.summary)
        write_csv(self.output/'episodes.csv', self.results)
        print('[EPISODE]', json.dumps(self.results[-1]), flush=True)
        self.physics[i], self.control[i] = [], []
        self.completed[i] += 1

    def finish(self):
        if not self.enabled:
            return
        for i in range(self._env.num_envs):
            if self.physics[i]:
                s = self._env.getup_state
                self._save(i, False, bool(s.success[i]), int(s.success_steps[i]), ['interrupted'])
        self.enabled = False
        if self.plots:
            for row in self.results:
                with np.load(self.output/row['file']) as data:
                    plot_episode(self.output/'plots'/Path(row['file']).stem, data, self.names,
                                 {k: v[row['env_id']] for k, v in self.limits.items()}, self._env.cfg.pose.success_height)
