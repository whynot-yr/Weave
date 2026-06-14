"""Replay ALL retargeted G1 trajectories of ONE object (a dict-of-trajectories pkl, e.g.
``data/retargeted/smalltable_nonlift20.pkl``) and pack them into a SINGLE packed ``.npz`` for
multi-clip RL training.

Processes one object per launch: spawns that object in the scene and drives it kinematically every
frame (so you can open the GUI and verify the robot/object interaction), then loops over every
trajectory and writes them all into one packed npz.

Packed npz layout (read by the training-side MotionLoader)::

    fps:            (1,)        int
    motion_lengths: (M,)        int64   # frames per clip
    object_names:   (M,)        str     # object per clip (all equal for one pkl)
    motion_names:   (M,)        str     # trajectory id per clip (debugging)
    <frame field>:  (sum_T, ..)         # every per-frame field concatenated along axis 0

.. code-block:: bash

    python ./scripts/data_replay_multiple.py \
        --input_file ./data/retargeted/smalltable_nonlift20.pkl \
        --output_file ./data/example_data/smalltable_20clip.npz --input_fps 30 --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay all trajectories of one object and pack them into a single npz.")
parser.add_argument("--input_file", type=str, required=True, help="A dict-of-trajectories pkl for ONE object.")
out = parser.add_mutually_exclusive_group(required=True)
out.add_argument("--output_file", type=str, default=None, help="Pack ALL trajectories into ONE npz at this path.")
out.add_argument("--output_dir", type=str, default=None, help="Write the packed npz into this dir, named <object>.npz.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input motion.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the output motion (must match training sim).")
parser.add_argument("--min_output_frames", type=int, default=2, help="Skip trajectories shorter than this many output frames.")
parser.add_argument("--limit", type=int, default=None, help="Process at most this many trajectories (debug).")
parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite an existing output npz.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import torch
import joblib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from dataclasses import MISSING
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_matrix, quat_mul, quat_slerp, quat_unique

from g1_hoi_learning.robots.g1_inspire import G1_INSPIRE_CFG
from g1_hoi_learning.objects.object_cfg import OBJECT_CFG_BY_NAME

# Joint ordering from the pytorch-kinematics URDF parse (matches data["joint_pos"] columns).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint", "L_thumb_intermediate_joint", "L_thumb_distal_joint",
    "L_index_proximal_joint", "L_index_intermediate_joint", "L_middle_proximal_joint", "L_middle_intermediate_joint",
    "L_ring_proximal_joint", "L_ring_intermediate_joint", "L_pinky_proximal_joint", "L_pinky_intermediate_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    "R_index_proximal_joint", "R_index_intermediate_joint", "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_ring_proximal_joint", "R_ring_intermediate_joint", "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    """Replay scene with the robot AND the object (resolved per-pkl in main, visible in the GUI)."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    obj: RigidObjectCfg = MISSING  # set in main() from the pkl's object_name


class MotionLoader:
    """Interpolate one trajectory from input_fps to output_fps and compute velocities (matches data_replay.py)."""

    def __init__(self, data: dict, input_fps: int, output_fps: int, device: torch.device):
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion(data)
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self, data: dict):
        self.root_poss_input = torch.from_numpy(np.asarray(data["root_trans"])).to(self.device, dtype=torch.float32)
        self.root_rots_input = torch.from_numpy(np.asarray(data["root_quat"])).to(self.device, dtype=torch.float32)
        self.dof_poss_input = torch.from_numpy(np.asarray(data["joint_pos"])).to(self.device, dtype=torch.float32)
        obj = data["object"]
        self.object_poss_input = torch.from_numpy(np.asarray(obj["trans"])).to(self.device, dtype=torch.float32)
        self.object_rots_input = quat_unique(
            quat_from_matrix(torch.from_numpy(np.asarray(obj["rot"])).to(self.device, dtype=torch.float32))
        )
        self.contact_label_input = torch.from_numpy(np.asarray(data["contact_label"])).to(self.device, dtype=torch.float32)
        self.input_frames = self.root_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        idx0, idx1, blend = self._compute_frame_blend(times)
        self.root_poss = self._lerp(self.root_poss_input[idx0], self.root_poss_input[idx1], blend.unsqueeze(1))
        self.root_rots = self._slerp(self.root_rots_input[idx0], self.root_rots_input[idx1], blend)
        self.dof_poss = self._lerp(self.dof_poss_input[idx0], self.dof_poss_input[idx1], blend.unsqueeze(1))
        self.object_poss = self._lerp(self.object_poss_input[idx0], self.object_poss_input[idx1], blend.unsqueeze(1))
        self.object_rots = self._slerp(self.object_rots_input[idx0], self.object_rots_input[idx1], blend)
        nearest = torch.where(blend < 0.5, idx0, idx1)
        self.contact_labels = self.contact_label_input[nearest]

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - idx0
        return idx0, idx1, blend

    def _compute_velocities(self):
        self.root_lin_vels = torch.gradient(self.root_poss, spacing=self.output_dt, dim=0)[0]
        self.dof_vels = torch.gradient(self.dof_poss, spacing=self.output_dt, dim=0)[0]
        self.root_ang_vels = self._so3_derivative(self.root_rots, self.output_dt)
        self.object_lin_vels = torch.gradient(self.object_poss, spacing=self.output_dt, dim=0)[0]
        self.object_ang_vels = self._so3_derivative(self.object_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        i = self.current_idx
        state = (
            self.root_poss[i : i + 1], self.root_rots[i : i + 1], self.root_lin_vels[i : i + 1], self.root_ang_vels[i : i + 1],
            self.dof_poss[i : i + 1], self.dof_vels[i : i + 1],
            self.object_poss[i : i + 1], self.object_rots[i : i + 1], self.object_lin_vels[i : i + 1], self.object_ang_vels[i : i + 1],
            self.contact_labels[i],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag:
            self.current_idx = 0
        return state, reset_flag


def compute_reorder_idx(robot, link_names: list[str]) -> list[int]:
    """Map robot.body_names (54) -> column index in the pkl's link_names (L). Errors loudly on any miss."""
    pk = {name: i for i, name in enumerate(link_names)}
    missing = [name for name in robot.body_names if name not in pk]
    if missing:
        raise KeyError(f"{len(missing)} robot body name(s) not in pkl link_names: {missing}")
    return [pk[name] for name in robot.body_names]


def replay_trajectory(sim, scene, joint_indices, reorder_idx, traj_data: dict) -> dict:
    """Replay one trajectory (robot + object driven kinematically) -> log dict."""
    traj_data = dict(traj_data)
    traj_data["contact_label"] = np.asarray(traj_data["contact_label"])[:, reorder_idx]
    motion = MotionLoader(traj_data, args_cli.input_fps, args_cli.output_fps, sim.device)
    robot = scene["robot"]
    obj_asset = scene["obj"]
    realtime = not args_cli.headless
    frame_dt = 1.0 / args_cli.output_fps

    log = {
        "fps": [args_cli.output_fps], "object_name": str(traj_data["object"]["name"]),
        "joint_pos": [], "joint_vel": [],
        "body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": [],
        "object_pos_w": [], "object_quat_w": [], "object_lin_vel_w": [], "object_ang_vel_w": [],
        "contact_label": [],
    }

    while simulation_app.is_running():
        t0 = time.time()
        (
            (root_pos, root_rot, root_lin_vel, root_ang_vel, dof_pos, dof_vel,
             obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, contact_label),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = root_rot
        root_states[:, 7:10] = root_lin_vel
        root_states[:, 10:] = root_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indices] = dof_pos
        joint_vel[:, joint_indices] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        obj_state = obj_asset.data.default_root_state.clone()
        obj_state[:, :3] = obj_pos
        obj_state[:, :2] += scene.env_origins[:, :2]
        obj_state[:, 3:7] = obj_rot
        obj_state[:, 7:10] = obj_lin_vel
        obj_state[:, 10:] = obj_ang_vel
        obj_asset.write_root_state_to_sim(obj_state)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())
        log["object_pos_w"].append(obj_asset.data.body_pos_w[0, 0].cpu().numpy().copy())
        log["object_quat_w"].append(obj_asset.data.body_quat_w[0, 0].cpu().numpy().copy())
        log["object_lin_vel_w"].append(obj_asset.data.body_lin_vel_w[0, 0].cpu().numpy().copy())
        log["object_ang_vel_w"].append(obj_asset.data.body_ang_vel_w[0, 0].cpu().numpy().copy())
        log["contact_label"].append(contact_label.cpu().numpy().copy())

        if realtime:
            remaining = frame_dt - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

        if reset_flag:
            for k in list(log.keys()):
                if k not in ("fps", "object_name"):
                    log[k] = np.stack(log[k], axis=0)
            break
    return log


def iter_trajectories(pkl_path: str):
    """Yield (traj_id, traj_dict). Supports both a dict-of-trajectories pkl and a flat single-trajectory pkl."""
    data = joblib.load(pkl_path)
    if "root_trans" in data:  # flat single-trajectory pkl
        yield os.path.splitext(os.path.basename(pkl_path))[0], data
        return
    for traj_id, traj in data.items():
        yield str(traj_id), traj


FRAME_KEYS = [
    "joint_pos", "joint_vel",
    "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w",
    "contact_label",
]


def pack_and_save(logs: list[dict], out_path: str) -> None:
    packed = {k: np.concatenate([lg[k] for lg in logs], axis=0) for k in FRAME_KEYS}
    packed["motion_lengths"] = np.array([int(lg["joint_pos"].shape[0]) for lg in logs], dtype=np.int64)
    packed["object_names"] = np.array([str(lg["object_name"]) for lg in logs])
    packed["motion_names"] = np.array([str(lg["motion_name"]) for lg in logs])
    packed["fps"] = np.array([int(args_cli.output_fps)], dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **packed)
    print(f"  [save] {len(logs)} clips, {int(packed['motion_lengths'].sum())} frames -> {out_path}")


def main():
    pkl_path = args_cli.input_file
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"input_file not found: {pkl_path}")

    first_traj = next(iter(iter_trajectories(pkl_path)))[1]
    object_name = str(first_traj["object"]["name"])
    if object_name not in OBJECT_CFG_BY_NAME:
        raise KeyError(f"Unknown object_name '{object_name}'; not in OBJECT_CFG_BY_NAME")
    print(f"[INFO]: object = {object_name}")

    if args_cli.output_file is not None:
        out_path = args_cli.output_file
    else:
        os.makedirs(args_cli.output_dir, exist_ok=True)
        out_path = os.path.join(args_cli.output_dir, f"{object_name}.npz")
    if os.path.isfile(out_path) and not args_cli.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace.")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.obj = OBJECT_CFG_BY_NAME[object_name].replace(prim_path="{ENV_REGEX_NS}/Object")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete. Replay scene (robot + object) ready.")

    robot = scene["robot"]
    joint_indices, _ = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)

    logs: list[dict] = []
    reorder_idx = None
    count = 0
    for traj_id, traj in iter_trajectories(pkl_path):
        if args_cli.limit is not None and count >= args_cli.limit:
            break
        count += 1
        if str(traj["object"]["name"]) != object_name:
            print(f"  [skip] {traj_id}: object {traj['object']['name']!r} != {object_name!r}")
            continue
        if reorder_idx is None:
            reorder_idx = compute_reorder_idx(robot, list(traj["link_names"]))
        in_frames = int(np.asarray(traj["joint_pos"]).shape[0])
        log = replay_trajectory(sim, scene, joint_indices, reorder_idx, traj)
        log["motion_name"] = str(traj_id)
        out_frames = int(log["joint_pos"].shape[0])
        if out_frames < args_cli.min_output_frames:
            print(f"  [skip] {traj_id}: {out_frames} < min_output_frames={args_cli.min_output_frames}")
            continue
        logs.append(log)
        print(f"  [ok]   {traj_id}: {in_frames} in -> {out_frames} out frames")

    if not logs:
        print("[WARN]: no trajectories produced; nothing written.")
        return
    pack_and_save(logs, out_path)
    print(f"\n[INFO]: Done. {len(logs)} clips of '{object_name}' -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
