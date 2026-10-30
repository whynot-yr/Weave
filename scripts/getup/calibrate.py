"""Inspect candidate getup poses with WEAVE assets; no policy or training changes."""
import argparse
import csv
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pose", choices=["standing", "supine"], default="standing")
parser.add_argument("--root-height", type=float, default=None)
parser.add_argument("--head-offset", type=float, nargs=3, default=None)
parser.add_argument("--pose-json", type=Path, help="Optional root_pos/root_quat_wxyz/joint_pos overrides")
parser.add_argument("--randomize", action="store_true", help="HoST joint perturbations; enable after checking nominal pose")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--hold-pose", action="store_true", help="Rewrite initial state every step for geometry inspection; NOT dynamics validation")
parser.add_argument("--output", type=Path, default=Path("outputs/getup_calibration"))
parser.add_argument("--profile", action="store_true", help="Synchronize CUDA and report per-section wall time")
parser.add_argument("--no-marker", action="store_true", help="Diagnostic: skip head marker USD updates")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
startup_time = time.perf_counter()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from g1_hoi_learning.robots.g1_inspire import G1_INSPIRE_CFG
from g1_hoi_learning.assets import GROUND_PLANE_USD_PATH
from g1_hoi_learning.tasks.hoi.env_cfg import ActionsCfg
from g1_hoi_learning.tasks.getup.pose_cfg import GetupPoseCfg


@configclass
class CalibrationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path=GROUND_PLANE_USD_PATH,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    torch.manual_seed(args.seed)
    if args.head_offset is None:
        args.head_offset = GetupPoseCfg().head_offset
    args.output.mkdir(parents=True, exist_ok=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=4, device=args.device)
    sim_cfg.physx.gpu_max_rigid_patch_count = 16 * 2**16
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.6], [0.0, 0.0, 0.45])
    scene = InteractiveScene(CalibrationSceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    robot = scene["robot"]
    root = robot.data.default_root_state.clone()
    q = robot.data.default_joint_pos.clone()
    # Explicit provisional candidate: WEAVE joint pose rotated onto its back.
    # Height/pose are NOT promoted to training defaults by this script.
    if args.pose == "supine":
        pose_cfg = GetupPoseCfg()
        root[:, :3] = torch.tensor(pose_cfg.root_pos, device=sim.device)
        root[:, 3:7] = torch.tensor(pose_cfg.root_quat, device=sim.device)
    overrides = json.loads(args.pose_json.read_text()) if args.pose_json else {}
    if "root_pos" in overrides:
        root[:, :3] = torch.tensor(overrides["root_pos"], device=sim.device)
    if "root_quat_wxyz" in overrides:
        quat = torch.tensor(overrides["root_quat_wxyz"], device=sim.device, dtype=root.dtype)
        if quat.shape != (4,) or not torch.isfinite(quat).all() or quat.norm() < 1e-6:
            raise ValueError("Invalid root quaternion")
        root[:, 3:7] = quat / quat.norm()
    if args.root_height is not None:
        root[:, 2] = args.root_height
    root[:, :3] += scene.env_origins
    root[:, 7:] = 0
    for name, value in overrides.get("joint_pos", {}).items():
        q[:, robot.joint_names.index(name)] = float(value)
    mimic = ActionsCfg().joint_pos.mimic
    drive_ids = [i for i, name in enumerate(robot.joint_names) if name not in mimic]
    assert (robot.num_joints, len(drive_ids)) == (53, 41), robot.joint_names
    limits = robot.data.joint_pos_limits[0].clone()
    # Restrict driver samples so derived mimic states also satisfy hard limits.
    for name, (driver, multiplier, offset) in mimic.items():
        j, d = robot.joint_names.index(name), robot.joint_names.index(driver)
        lo, hi = sorted(((limits[j, 0].item()-offset)/multiplier, (limits[j, 1].item()-offset)/multiplier))
        limits[d, 0] = max(limits[d, 0].item(), lo)
        limits[d, 1] = min(limits[d, 1].item(), hi)
    if (limits[:, 0] > limits[:, 1]).any():
        raise ValueError("Inconsistent mimic limits")
    if args.randomize:
        x = q[:, drive_ids]
        q[:, drive_ids] = x * (0.9 + 0.2 * torch.rand_like(x)) + (0.2 * torch.rand_like(x) - 0.1)
    q[:, drive_ids] = q[:, drive_ids].clamp(limits[drive_ids, 0], limits[drive_ids, 1])
    for name, (driver, multiplier, offset) in mimic.items():
        q[:, robot.joint_names.index(name)] = q[:, robot.joint_names.index(driver)] * multiplier + offset
    qd = torch.zeros_like(q)

    def reset_state():
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(q, qd)
        robot.set_joint_position_target(q)

    reset_state()
    scene.reset()
    body_ids = {name: robot.body_names.index(name) for name in
                ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]}
    marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/HeadKeypoint")
    marker_cfg.markers["sphere"].radius = 0.025
    marker = VisualizationMarkers(marker_cfg)
    offset = torch.tensor([args.head_offset], device=sim.device)
    manifest = {
        "candidate_only": True, "pose": args.pose, "hold_pose": args.hold_pose,
        "randomize": args.randomize, "seed": args.seed,
        "root_pos": (root[0, :3] - scene.env_origins[0]).tolist(),
        "root_quat_wxyz": root[0, 3:7].tolist(),
        "joint_pos": dict(zip(robot.joint_names, q[0].tolist())),
        "head_offset": args.head_offset, "body_names": robot.body_names,
        "note": "Head offset is a COM-derived candidate. Body origins do not certify collision clearance.",
    }
    (args.output / "candidate.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    timings = {key: [] for key in ["write", "physics", "scene_update", "body_read", "marker", "metrics"]}

    def stamp():
        if args.profile and str(sim.device).startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter()

    print(f"[TIMING] Startup/setup: {stamp()-startup_time:.3f}s", flush=True)
    loop_start = stamp()
    for step in range(round(args.seconds / 0.005)):
        if not app.is_running():
            break
        start = stamp()
        if args.hold_pose:
            reset_state()
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        end = stamp()
        timings["write"].append(end-start)
        sim.step(render=not args.headless and step % 4 == 0)
        start = stamp()
        timings["physics"].append(start-end)
        scene.update(0.005)
        end = stamp()
        timings["scene_update"].append(end-start)
        pos, quat = robot.data.body_link_pos_w, robot.data.body_link_quat_w
        torso = body_ids["torso_link"]
        head = pos[:, torso] + quat_apply(quat[:, torso], offset)
        start = stamp()
        timings["body_read"].append(start-end)
        if not args.no_marker:
            marker.visualize(head)
        end = stamp()
        timings["marker"].append(end-start)
        if step % 4 == 0:
            left = pos[0, body_ids["left_ankle_roll_link"], 2].item()
            right = pos[0, body_ids["right_ankle_roll_link"], 2].item()
            row = dict(time_s=(step+1)*0.005,
                       pelvis_height=pos[0, body_ids["pelvis"], 2].item(),
                       head_height=head[0, 2].item(), left_foot_height=left, right_foot_height=right,
                       head_relative_feet=head[0, 2].item()-(left+right)/2,
                       base_speed=robot.data.root_lin_vel_w[0].norm().item(),
                       max_joint_speed=robot.data.joint_vel[0].abs().max().item())
            rows.append(row)
            if step % 200 == 0:
                print("[CALIBRATION]", row, flush=True)
        timings["metrics"].append(stamp()-end)
        if args.profile and step % 10 == 0:
            print("[PROFILE]", step, {k: round(v[-1]*1000, 3) for k,v in timings.items()}, flush=True)
    profile = {k: {"total_s": sum(v), "mean_ms": 1000*sum(v)/max(len(v),1),
                   "max_ms": 1000*max(v, default=0)} for k,v in timings.items()}
    profile["loop_wall_s"] = stamp()-loop_start
    profile["cuda_synchronized"] = args.profile
    (args.output / "timing.json").write_text(json.dumps(profile, indent=2))
    print("[TIMING]", profile, flush=True)
    if rows:
        with (args.output / "measurements.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[CALIBRATION] Saved candidate and measurements to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Unsubscribe Isaac Lab's stop callback before closing the Kit stage.
        # Otherwise standalone shutdown can enter its "keep rendering" loop.
        sim_utils.SimulationContext.clear_instance()
        app.close(wait_for_replicator=False)
