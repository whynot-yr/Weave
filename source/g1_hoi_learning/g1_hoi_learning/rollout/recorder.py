from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from g1_hoi_learning.tasks.hoi.mdp.actions import MimicJointPositionAction
from g1_hoi_learning.tasks.hoi.mdp.commands import MotionCommand

from .writer import RolloutH5Writer


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@dataclass
class PendingStep:
    env_ids: torch.Tensor
    step: int
    values: dict[str, np.ndarray]


class RolloutRecorder:
    """Capture state/action before a step and attach its outcome after the step."""

    def __init__(
        self,
        env,
        output_dir: str,
        *,
        initial_obs,
        clip_names: list[str],
        selection_seed: int,
        environment_seed: int,
        reference_file: str,
        checkpoint: str,
        contact_threshold: float = 1.0,
    ):
        self.wrapper = env
        self.env = env.unwrapped
        self.command: MotionCommand = self.env.command_manager.get_term("motion")
        self.robot = self.env.scene["robot"]
        self.contact_sensor = self.env.scene["contact_sensor"]
        self.camera = self.env.scene["head_camera"]
        self.action_term: MimicJointPositionAction = self.env.action_manager.get_term("joint_pos")
        self.contact_threshold = float(contact_threshold)
        self.num_envs = self.env.num_envs
        self.active = torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device)
        self.lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        self.causes = [""] * self.num_envs

        clip_ids = _numpy(self.command.env_clip).astype(np.int32)
        ref_lengths = _numpy(self.command.motion.clip_lengths[self.command.env_clip]).astype(np.int32)
        selected_names = [clip_names[int(i)] for i in clip_ids]
        object_names = [self.command.motion.object_names[int(self.command.motion.clip_object[i])] for i in clip_ids]
        max_steps = int(ref_lengths.max()) + 5
        reward_names = list(self.env.reward_manager.active_terms)

        metadata = {
            "reference_file": reference_file,
            "checkpoint": checkpoint,
            "sampling_seed": int(selection_seed),
            "environment_seed": int(environment_seed),
            "sampling_strategy": "uniform_without_replacement",
            "fps": float(1.0 / self.env.step_dt),
            "dt": float(self.env.step_dt),
            "camera": {
                "name": "head_camera",
                "width": int(self.camera.cfg.width),
                "height": int(self.camera.cfg.height),
                "data_types": list(self.camera.cfg.data_types),
                "depth_type": "distance_to_image_plane",
                "depth_unit": "millimeter",
                "clipping_range_m": list(self.camera.cfg.spawn.clipping_range),
            },
            "joint_names": list(self.robot.joint_names),
            "body_names": list(self.robot.body_names),
            "action_names": list(self.action_term._joint_names),
            "observation_groups": list(initial_obs.keys()),
            "reward_terms": reward_names,
            "contact_threshold_n": self.contact_threshold,
        }
        self.metadata = metadata
        self.writer = RolloutH5Writer(
            output_dir,
            clip_ids=clip_ids,
            clip_names=selected_names,
            reference_lengths=ref_lengths,
            object_names=object_names,
            episode_seed=environment_seed,
            max_steps=max_steps,
            metadata=metadata,
        )

    @property
    def done(self) -> bool:
        return bool((~self.active).all())

    @property
    def max_steps(self) -> int:
        return self.writer.max_steps

    def capture_before_step(self, obs, policy_actions: torch.Tensor) -> PendingStep:
        env_ids = self.active.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            raise RuntimeError("capture_before_step called after every episode finished")
        active_lengths = self.lengths[env_ids]
        if not torch.all(active_lengths == active_lengths[0]):
            raise RuntimeError("V1 collector expects all active environments to start together")
        step = int(active_lengths[0])
        origin = self.env.scene.env_origins

        applied_actions = policy_actions
        if self.wrapper.clip_actions is not None:
            applied_actions = torch.clamp(policy_actions, -self.wrapper.clip_actions, self.wrapper.clip_actions)
        processed_actions, all_targets = self.action_term.expand_actions(applied_actions)

        forces = self.contact_sensor.data.force_matrix_w[:, 0, :, :]
        force_norm = torch.linalg.norm(forces, dim=-1)
        camera_rgb = self.camera.data.output["rgb"]
        camera_depth = self.camera.data.output["depth"].squeeze(-1)
        near, far = self.camera.cfg.spawn.clipping_range
        depth_valid = torch.isfinite(camera_depth) & (camera_depth >= near) & (camera_depth <= far)
        depth_mm = torch.where(
            depth_valid,
            torch.clamp(torch.round(camera_depth * 1000.0), 1, 65535),
            torch.zeros_like(camera_depth),
        ).to(torch.int32)

        root = self.robot.data
        values: dict[str, torch.Tensor] = {
            "frame/step": torch.full((self.num_envs,), step, dtype=torch.int32, device=self.env.device),
            "frame/ref_step": self.command.time_steps.to(torch.int32),
            "frame/sim_time": torch.full(
                (self.num_envs,), step * self.env.step_dt, dtype=torch.float32, device=self.env.device
            ),
            "robot/joint_pos": self.command.robot_joint_pos,
            "robot/joint_vel": self.command.robot_joint_vel,
            "robot/joint_effort": root.applied_torque,
            "robot/joint_pos_target": all_targets,
            "robot/root_pos": root.root_link_pos_w - origin,
            "robot/root_quat": root.root_link_quat_w,
            "robot/root_lin_vel": root.root_link_lin_vel_w,
            "robot/root_ang_vel": root.root_link_ang_vel_w,
            "robot/body_pos": self.command.robot_body_pos_w - origin[:, None, :],
            "robot/body_quat": self.command.robot_body_quat_w,
            "robot/body_lin_vel": self.command.robot_body_lin_vel_w,
            "robot/body_ang_vel": self.command.robot_body_ang_vel_w,
            "object/pos": self.command.obj_pos_w - origin,
            "object/quat": self.command.obj_quat_w,
            "object/lin_vel": self.command.obj_lin_vel_w,
            "object/ang_vel": self.command.obj_ang_vel_w,
            "contact/force_w": forces,
            "contact/force_norm": force_norm,
            "contact/actual": force_norm > self.contact_threshold,
            "reference/contact_label": self.command.ref_contact_label.to(torch.int8),
            "reference/joint_pos": self.command.joint_pos,
            "reference/joint_vel": self.command.joint_vel,
            "reference/body_pos": self.command.body_pos_w - origin[:, None, :],
            "reference/body_quat": self.command.body_quat_w,
            "reference/body_lin_vel": self.command.body_lin_vel_w,
            "reference/body_ang_vel": self.command.body_ang_vel_w,
            "reference/object_pos": self.command.ref_obj_pos_w - origin,
            "reference/object_quat": self.command.ref_obj_quat_w,
            "reference/object_lin_vel": self.command.ref_obj_lin_vel_w,
            "reference/object_ang_vel": self.command.ref_obj_ang_vel_w,
            "policy/action_raw": policy_actions,
            "policy/action_applied": applied_actions,
            "policy/action_processed": processed_actions,
            "policy/action_all_joints": all_targets,
            "camera/head/rgb": camera_rgb,
            "camera/head/depth_mm": depth_mm,
            "camera/head/intrinsic": self.camera.data.intrinsic_matrices,
            "camera/head/position": self.camera.data.pos_w - origin,
            "camera/head/quaternion_world": self.camera.data.quat_w_world,
            "camera/head/quaternion_ros": self.camera.data.quat_w_ros,
            "camera/head/timestamp": torch.full(
                (self.num_envs,), step * self.env.step_dt, dtype=torch.float64, device=self.env.device
            ),
        }
        for name in obs.keys():
            value = obs[name]
            if isinstance(value, torch.Tensor):
                values[f"policy/observation/{name}"] = value

        selected = {name: _numpy(value[env_ids]) for name, value in values.items()}
        selected["camera/head/depth_mm"] = selected["camera/head/depth_mm"].astype(np.uint16)
        return PendingStep(env_ids=env_ids, step=step, values=selected)

    def capture_after_step(self, pending: PendingStep, rewards: torch.Tensor) -> None:
        env_ids = pending.env_ids
        terminated = self.env.reset_terminated[env_ids].bool()
        truncated = self.env.reset_time_outs[env_ids].bool()
        done = terminated | truncated
        pending.values["transition/reward"] = _numpy(rewards[env_ids].to(torch.float32))
        pending.values["transition/terminated"] = _numpy(terminated)
        pending.values["transition/truncated"] = _numpy(truncated)
        pending.values["transition/done"] = _numpy(done)
        pending.values["transition/reward_terms"] = _numpy(self.env.reward_manager._step_reward[env_ids])
        self.writer.write_step(_numpy(env_ids), pending.step, pending.values)
        self.lengths[env_ids] += 1

        if not done.any():
            return
        term_mgr = self.env.termination_manager
        flags = {name: term_mgr.get_term(name) for name in term_mgr.active_terms}
        fail_names = [name for name in term_mgr.active_terms if not term_mgr.get_term_cfg(name).time_out]
        for local_index in done.nonzero(as_tuple=False).flatten().tolist():
            env_id = int(env_ids[local_index])
            is_success = bool(flags["clip_end"][env_id]) and not bool(terminated[local_index])
            if is_success:
                cause = "clip_end"
            elif bool(terminated[local_index]):
                cause = next((name for name in fail_names if bool(flags[name][env_id])), "unknown")
            else:
                cause = "time_out"
            self.active[env_id] = False
            self.success[env_id] = is_success
            self.causes[env_id] = cause
            self.writer.finish_episode(
                env_id, length=int(self.lengths[env_id]), success=is_success, reason=cause
            )

    def close(self) -> None:
        # Preserve partial trajectories if collection stopped before all envs finished.
        for env_id in self.active.nonzero(as_tuple=False).flatten().tolist():
            self.writer.finish_episode(
                env_id, length=int(self.lengths[env_id]), success=False, reason="sim_closed"
            )
            self.causes[env_id] = "sim_closed"
        result = {
            **self.metadata,
            "num_episodes": self.num_envs,
            "num_frames": int(self.lengths.sum()),
            "num_success": int(self.success.sum()),
            "success_rate": float(self.success.float().mean()),
            "termination_counts": {
                cause: self.causes.count(cause) for cause in sorted(set(self.causes)) if cause
            },
            "selected_clip_ids": _numpy(self.command.env_clip).astype(int).tolist(),
        }
        self.writer.close(result)
