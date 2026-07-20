"""Benchmark MultiMeshRaycasterV2 depth raycasting: per-step time + peak VRAM vs mesh count.

Cost is O(N * n_meshes * n_rays): the fused kernel launches dim=(N, n_meshes, n_rays) and
materializes a (N, n_meshes, n_rays) intermediate before the min-over-mesh reduction. So we time
two real endpoints -- full (robot+object+ground ~56) and min (object+ground ~2) -- and extrapolate
intermediate mesh counts linearly (exact, since the launch grid is linear in n_meshes).

Usage:
  ../sim51/bin/python scripts/benchmark_raycast.py --num_envs 1024 --headless
"""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Raycast depth benchmark.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--rays_w", type=int, default=80)
parser.add_argument("--rays_h", type=int, default=60)
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--max_dist", type=float, default=5.0)
parser.add_argument("--task", type=str, default="G1-Inspire-HOI-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gc

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import g1_hoi_learning.tasks  # noqa: F401
from simple_raycaster.raycaster_v2 import MultiMeshRaycasterV2

# D435i (same as tasks/depth/mdp/depth.py)
CAM_POS = (0.0576235, 0.01753, 0.42987)
CAM_RPY = (0.0, 0.8307767239493009, 0.0)
FOV_H, FOV_V = 86.0, 57.0


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    # settle poses with a few zero-action steps
    act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    for _ in range(3):
        env.step(act)

    dev = env.device
    dev_idx = torch.device(dev).index or 0
    n_env = env.num_envs
    w, h = args_cli.rays_w, args_cli.rays_h
    n_rays = w * h
    max_dist = args_cli.max_dist

    # pinhole ray grid (camera frame: forward +x, image-right +y, image-up +z)
    fh, fv = math.radians(FOV_H), math.radians(FOV_V)
    u = torch.linspace(math.tan(fh / 2), -math.tan(fh / 2), w, device=dev)
    v = torch.linspace(math.tan(fv / 2), -math.tan(fv / 2), h, device=dev)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    dirs = torch.stack([torch.ones_like(uu), uu, vv], dim=-1).reshape(-1, 3)
    dirs = (dirs / dirs.norm(dim=-1, keepdim=True)).contiguous()
    cam_pos_t = torch.tensor(CAM_POS, device=dev)
    cam_quat_t = quat_from_euler_xyz(*(torch.tensor(a, device=dev) for a in CAM_RPY)).reshape(4)
    robot = env.scene["robot"]
    torso = robot.find_bodies("torso_link")[0][0]

    def compute(rc):
        tp = robot.data.body_pos_w[:, torso]
        tq = robot.data.body_quat_w[:, torso]
        cp = tp + quat_apply(tq, cam_pos_t.expand(n_env, 3))
        cq = quat_mul(tq, cam_quat_t.expand(n_env, 4))
        rd = quat_apply(cq[:, None, :].expand(n_env, n_rays, 4), dirs[None].expand(n_env, n_rays, 3))
        rs = cp[:, None, :].expand(n_env, n_rays, 3).contiguous()
        _, dist = rc.raycast_fused(rs, rd.contiguous(), min_dist=0.02, max_dist=max_dist)
        return (dist / max_dist).clamp(0.0, 1.0)

    def build_full(rc):
        rc.add_isaac_entity(env.scene["robot"])
        try:
            rc.add_isaac_entity(env.scene["object"])
        except Exception as e:  # noqa: BLE001  RigidObject registration is untested in the lib
            print(f"  [warn] object registration failed, skipping: {e}")
        rc.add_isaac_static("/World/ground")

    def build_min(rc):
        try:
            rc.add_isaac_entity(env.scene["object"])
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] object registration failed, skipping: {e}")
        rc.add_isaac_static("/World/ground")

    def bench(name, build):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free0, _ = torch.cuda.mem_get_info(dev_idx)
        torch.cuda.reset_peak_memory_stats()
        rc = MultiMeshRaycasterV2(dev)
        build(rc)
        for _ in range(args_cli.warmup):  # compile warp kernels + build BVH
            compute(rc)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args_cli.iters):
            compute(rc)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args_cli.iters
        peak = torch.cuda.max_memory_allocated(dev_idx)
        free1, _ = torch.cuda.mem_get_info(dev_idx)
        m = rc.n_meshes
        inter = n_env * m * n_rays * 4
        print(
            f"[{name:5s}] meshes={m:3d}  {dt * 1e3:7.2f} ms/step  {1 / dt:7.1f} Hz  "
            f"torch_peak={peak / 1e9:6.3f} GB  gpu_delta={(free0 - free1) / 1e9:6.3f} GB  "
            f"intermediate(N*m*R*4)={inter / 1e9:6.3f} GB"
        )
        del rc
        gc.collect()
        torch.cuda.empty_cache()
        return m, dt

    print(
        f"\n=== raycast benchmark: N={n_env}  rays={w}x{h}={n_rays}  "
        f"iters={args_cli.iters}  device={dev} ===\n"
    )
    m_full, t_full = bench("full", build_full)
    m_min, t_min = bench("min", build_min)

    # exact linear model: grid dim (N, n_meshes, n_rays) is linear in n_meshes
    if m_full != m_min:
        slope = (t_full - t_min) / (m_full - m_min)
        print("\n--- linear extrapolation (grid is linear in n_meshes) ---")
        for m in (4, 8, 16, 32):
            t = t_min + slope * (m - m_min)
            inter = n_env * m * n_rays * 4
            print(
                f"[extrap] meshes={m:3d}  {t * 1e3:7.2f} ms/step  {1 / t:7.1f} Hz  "
                f"intermediate={inter / 1e9:6.3f} GB"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
