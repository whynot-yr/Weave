"""Visualize what DeFM (ResNet-18) encodes from our noised depth, per env / per object.

Loads several DIFFERENT objects across envs (round-robin: env i -> object i % O). For each env it:
  * builds the NOISED metric depth the same way the obs DR does (distance^2 stereo noise + pixel
    dropout), but keeps it in METERS with holes/miss -> 0 (DeFM's invalid encoding);
  * runs it through frozen DeFM ResNet-18 (leggedrobotics/defm), reads the BiFPN P4 spatial token map
    (N,128,6,8), and PCA-reduces the 128-D features to RGB (joint PCA across all envs so colors are
    comparable) -- the standard DINO-style feature visualization.
Each grid cell = noised depth (top, turbo) over DeFM-P4 PCA-RGB (bottom). If DeFM has "understood" the
depth, object / ground / robot should show up as distinct feature-colour regions; a uniform blob means
its frozen features are off-distribution on our small sim raycast depth.

Usage:
  cd /home/ubuntu/Desktop/IsaacSim51/IsaacLab
  ./isaaclab.sh -p ../g1_hoi_learning/scripts/check_depth_multiobject.py
  # options: --num_envs N  --dr {none,pos,rot,intr,all}  --defm_path <clone>  --out ./depth_check_out
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

_DEFM_DEFAULT = "/tmp/claude-1000/-home-ubuntu-Desktop-IsaacSim51-g1-hoi-learning/4d426403-acd3-449c-891d-6f83a5018969/scratchpad/defm"

parser = argparse.ArgumentParser(description="Visualize DeFM ResNet-18 features on noised depth.")
parser.add_argument("--motion_files", nargs="+", default=None, help="npz files (different objects). Default: all data/train/*.npz")
parser.add_argument("--num_envs", type=int, default=None, help="default = number of distinct objects (one env each)")
parser.add_argument("--steps", type=int, default=0, help="zero-action sim steps before capture (0 = the valid RSI reset pose)")
parser.add_argument("--frames", type=int, default=1)
parser.add_argument("--out", type=str, default="./depth_check_out")
parser.add_argument("--task", type=str, default="G1-Inspire-HOI-Depth-v0")
parser.add_argument("--dr", choices=["none", "pos", "rot", "intr", "all"], default="none",
                    help="which camera extrinsics/intrinsics DR to ENABLE (default none = nominal camera)")
parser.add_argument("--defm_path", type=str, default=_DEFM_DEFAULT, help="path to the cloned leggedrobotics/defm repo")
parser.add_argument("--target_size", type=int, default=0,
                    help="if >0, resize depth to this before DeFM (finer P4, closer to DeFM's ~224 training "
                         "res). 0 = native pad to 96x128. Try 224.")
parser.add_argument("--square", action="store_true",
                    help="resize to target_size x target_size (DeFM's native square format; distorts our 16:9). "
                         "Default: aspect-preserving (width=target_size).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import g1_hoi_learning.tasks  # noqa: F401

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _obj_name(path: str) -> str:
    return os.path.basename(path).split("_")[0]


def _load_defm(device):
    """Load frozen DeFM ResNet-18 + its batched preprocess (from the cloned repo)."""
    sys.path.append(args_cli.defm_path)
    from defm.model_factory import create_defm_model
    from defm.utils import preprocess_depth_batch

    model = create_defm_model("defm_resnet18", pretrained=True).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess_depth_batch


def _blur(d, ksize=5, sigma=1.0):
    """Depthwise Gaussian blur (stereo smoothing; also yields edge flying-pixels). d: (N,H,W)."""
    ax = torch.arange(ksize, device=d.device, dtype=d.dtype) - (ksize - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    ker = torch.outer(g, g)
    ker = (ker / ker.sum()).view(1, 1, ksize, ksize)
    x = torch.nn.functional.pad(d.unsqueeze(1), (ksize // 2,) * 4, mode="reflect")
    return torch.nn.functional.conv2d(x, ker)[:, 0]


def _edge_holes(d, valid, edge_thresh=0.05, drop_p=0.5):
    """Holes at valid<->valid depth discontinuities ONLY (clip/hole borders excluded). d: (N,H,W)."""
    edge = torch.zeros_like(d, dtype=torch.bool)
    bx = ((d[:, :, 1:] - d[:, :, :-1]).abs() > edge_thresh) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, 1:] |= bx
    edge[:, :, :-1] |= bx
    by = ((d[:, 1:, :] - d[:, :-1, :]).abs() > edge_thresh) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, 1:, :] |= by
    edge[:, :-1, :] |= by
    edge &= torch.rand_like(d) < drop_p
    return torch.nn.functional.max_pool2d(edge.float()[:, None], 3, 1, 1)[:, 0] > 0  # dilate -> blobby


def _blob_holes(d, max_dist, scale=12, base_p=0.01, range_p=0.06):
    """Spatially-correlated (blob) holes, more frequent far away. d: (N,H,W)."""
    N, H, W = d.shape
    low = torch.rand(N, 1, max(1, H // scale), max(1, W // scale), device=d.device)
    field = torch.nn.functional.interpolate(low, (H, W), mode="bilinear", align_corners=False)[:, 0]
    return field < (base_p + range_p * (d / max_dist).clamp(0, 1))


def _degrade_depth(d, max_dist, noise_k=(0.005, 0.015), dropout=0.0,
                   min_z=0.25, blur_sigma=1.0, edge=(0.05, 0.5), blob=(12, 0.01, 0.06)):
    """Crisp raycast depth (m) -> D435i-like depth: Gaussian blur + axial noise + structured holes.

    Holes (sub-MinZ, >max clip / no-hit, real depth edges, blobs, salt-pepper) -> 0. Returns (N,H,W) m.
    """
    valid = (d > min_z) & (d < max_dist - 1e-3)         # real, unclipped surface
    hole = ~valid                                       # sub-MinZ + clipped / no-hit
    hole |= _edge_holes(d, valid, *edge)                # real depth discontinuities (valid<->valid)
    hole |= _blob_holes(d, max_dist, *blob)             # range-dependent blobs
    if dropout > 0.0:
        hole |= torch.rand_like(d) < dropout            # residual salt-pepper
    out = _blur(d, sigma=blur_sigma)                    # stereo smoothing + flying-pixels
    kk = torch.empty(d.shape[0], 1, 1, device=d.device).uniform_(*noise_k)
    out = out + torch.randn_like(out) * (kk * out**2)   # axial noise, sigma ~ k*z^2
    return out.masked_fill(hole, 0.0).clamp(0.0, max_dist)


def main():
    # ---- resolve motion files (need several DIFFERENT objects) ----
    if args_cli.motion_files:
        motion_files = args_cli.motion_files
    else:
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train")
        motion_files = sorted(glob.glob(os.path.join(root, "*.npz")))
    if not motion_files:
        raise FileNotFoundError("No motion files found; pass --motion_files a.npz b.npz ...")
    distinct = list(dict.fromkeys(_obj_name(f) for f in motion_files))
    n_obj = len(distinct)
    num_envs = args_cli.num_envs or n_obj
    print(f"[INFO] {len(motion_files)} file(s), {n_obj} distinct object(s): {distinct}")

    # ---- build env ----
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.commands.motion.motion_files = list(motion_files)
    # camera extrinsics/intrinsics DR: zero all, re-enable only the --dr component (isolation).
    ext = env_cfg.events.randomize_camera_extrinsics.params
    intr = env_cfg.events.randomize_camera_intrinsics.params
    orig = {"pos": ext.get("pos_noise_std"), "rot": ext.get("rot_noise_std"),
            "focal": intr.get("focal_length_noise_std"), "aperture": intr.get("aperture_noise_std")}
    ext["pos_noise_std"] = ext["rot_noise_std"] = (0.0, 0.0, 0.0)
    intr["focal_length_noise_std"] = intr["aperture_noise_std"] = 0.0
    if args_cli.dr in ("pos", "all"):
        ext["pos_noise_std"] = orig["pos"]
    if args_cli.dr in ("rot", "all"):
        ext["rot_noise_std"] = orig["rot"]
    if args_cli.dr in ("intr", "all"):
        intr["focal_length_noise_std"], intr["aperture_noise_std"] = orig["focal"], orig["aperture"]
    print(f"[INFO] camera DR enabled: {args_cli.dr}")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    cam = env.scene.sensors["depth_cam"]
    if hasattr(cam, "set_robot"):
        cam.set_robot(env.scene["robot"])
    obj = env.scene["object"]
    max_d = float(cam.cfg.max_distance)
    act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

    obs_params = dict(getattr(env_cfg.observations.depth.depth, "params", {}) or {})
    min_dist = float(obs_params.get("min_dist", 0.1))
    max_dist = float(obs_params.get("max_dist", max_d))
    noise_k_range = tuple(obs_params.get("noise_k_range", (0.005, 0.015)))
    dropout_prob = float(obs_params.get("dropout_prob", 0.05))

    motion_term = env.command_manager.get_term("motion")
    env_object = getattr(motion_term, "env_object", None)
    labels = [distinct[(int(env_object[i]) if env_object is not None else i) % n_obj] for i in range(env.num_envs)]

    hw = cam.image_shape  # (H, W)
    print("[INFO] loading DeFM ResNet-18 ...")
    defm, preprocess = _load_defm(env.device)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import minmax_scale

    os.makedirs(args_cli.out, exist_ok=True)

    for f in range(args_cli.frames):
        for _ in range(args_cli.steps):
            env.step(act)

        # ---- degraded METRIC depth (DeFM input), meters; holes -> 0 (D435i invalid) ----
        depth_m = cam.data.output["distance_to_image_plane"].squeeze(-1).detach()  # (N,H,W) meters
        n = depth_m.shape[0]
        noised = _degrade_depth(depth_m, max_dist, noise_k_range, dropout_prob)

        # ---- DeFM ResNet-18 -> P4 spatial tokens -> joint PCA to RGB ----
        if args_cli.target_size > 0:
            if args_cli.square:  # DeFM's native square format (distorts 16:9)
                h_t = w_t = max(32, (args_cli.target_size // 32) * 32)
            else:  # aspect-preserving (w=target, h=/32)
                w_t = max(32, (args_cli.target_size // 32) * 32)
                h_t = max(32, round(w_t * hw[0] / hw[1] / 32) * 32)
            x = preprocess(noised.unsqueeze(1), target_size=(h_t, w_t), device=depth_m.device)
            if f == 0:
                print(f"[INFO] DeFM input resized to {(h_t, w_t)} -> P4 {(h_t // 16, w_t // 16)}")
        else:
            x = preprocess(noised.unsqueeze(1), cnn_padding=True, device=depth_m.device)  # (N,3,96,128)
        # DeFM twice: FP32 (reference) + FP16 autocast (deploy/training precision); shared PCA basis
        with torch.no_grad():
            p4_32 = defm(x)["dense_bifpn"]["P4"]  # (N,128,gh,gw)
            with torch.autocast(device_type=("cuda" if x.is_cuda else "cpu"), dtype=torch.float16):
                p4_16 = defm(x)["dense_bifpn"]["P4"]
        gh, gw = p4_32.shape[2], p4_32.shape[3]
        feat32 = p4_32.permute(0, 2, 3, 1).reshape(-1, p4_32.shape[1]).float().cpu().numpy()
        feat16 = p4_16.permute(0, 2, 3, 1).reshape(-1, p4_16.shape[1]).float().cpu().numpy()
        pca = PCA(n_components=3).fit(feat32)  # basis from FP32 -> fp32/fp16 colors directly comparable
        both = minmax_scale(np.concatenate([pca.transform(feat32), pca.transform(feat16)], axis=0))

        def _up(rgb):  # (N*gh*gw, 3) -> (N,H,W,3), nearest (keep the honest feature grid)
            t = torch.from_numpy(rgb.reshape(n, gh, gw, 3).astype(np.float32)).permute(0, 3, 1, 2)
            return torch.nn.functional.interpolate(t, size=hw, mode="nearest").permute(0, 2, 3, 1).numpy()

        pca_up_32, pca_up_16 = _up(both[: len(feat32)]), _up(both[len(feat32):])

        noised_disp = (noised / max_dist).clamp(0, 1).cpu().numpy()  # for display; holes(0)->dark

        # ---- stats table (to file; terminal is flooded by sim warnings) ----
        obj_dist = (obj.data.root_pos_w - cam.data.pos_w).norm(dim=-1).cpu().numpy()
        root_z = env.scene["robot"].data.root_pos_w[:, 2].cpu().numpy()
        m_np = depth_m.cpu().numpy()
        lines = [f"=== frame {f}: HxW={hw[0]}x{hw[1]} P4={gh}x{gw} maxd={max_d} [DR:{args_cli.dr}] ===",
                 f"{'env':>3} {'object':>13} {'rootZ':>6} {'objd':>6} {'d_mean':>7} {'hit%':>6} {'hole%':>6}"]
        for i in range(n):
            d = m_np[i]
            hole = float((noised_disp[i] == 0.0).mean() * 100.0)
            flag = "  <- all-near (buried/occluded?)" if float(d.mean()) < 0.2 else ""
            lines.append(f"{i:>3} {labels[i]:>13} {root_z[i]:>6.2f} {obj_dist[i]:>6.2f} "
                         f"{float(d.mean()):>7.3f} {100.0*(d < max_d-1e-3).mean():>5.1f}% {hole:>5.1f}%{flag}")
        table = "\n".join(lines)
        print("\n" + table)
        with open(os.path.join(args_cli.out, f"depth_stats_frame{f}.txt"), "w") as fp:
            fp.write(table + "\n")

        _save_defm(noised_disp, pca_up_32, pca_up_16, labels, obj_dist, f, args_cli.out)

    print(f"\n[INFO] images written to: {os.path.abspath(args_cli.out)}")
    env.close()


def _save_defm(noised_disp, pca_up_32, pca_up_16, labels, obj_dist, frame, out_dir):
    """Square grid; each cell = noised depth (top) / DeFM-P4 PCA fp32 (mid) / fp16 (bottom)."""
    n = noised_disp.shape[0]
    if not HAS_MPL:
        np.savez(os.path.join(out_dir, f"defm_frame{frame}.npz"),
                 noised=noised_disp, pca32=pca_up_32, pca16=pca_up_16)
        print("[WARN] matplotlib not available; saved arrays as .npz.")
        return
    turbo = plt.get_cmap("turbo")
    w = noised_disp.shape[2]
    sep = np.ones((2, w, 3))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 4.2 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        top = turbo(noised_disp[idx])[..., :3]  # (H,W,3) depth colormapped
        cell = np.concatenate([top, sep, pca_up_32[idx], sep, pca_up_16[idx]], axis=0)  # noised / fp32 / fp16
        ax.imshow(cell)
        ax.set_title(f"env{idx} {labels[idx]} d={obj_dist[idx]:.2f}\nnoised / P4 fp32 / P4 fp16", fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, f"defm_features_frame{frame}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[INFO] saved {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
