# g1_hoi_learning

Isaac Lab extension for **Unitree G1 + Inspire dexterous hands** human-object-interaction (HOI) motion imitation. A single PPO policy learns to mimic mocap-retargeted reference motions of the robot manipulating a household object (clothesstand, suitcase, monitor, ...) while satisfying contact constraints on the hands.

Built on top of Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + RSL-RL.

---

## Project Layout

```
g1_hoi_learning/
├── configs/
│   ├── train.yaml                # Hydra config: env / agent overrides for training
│   └── play.yaml                 # inherits train.yaml; num_envs=1, RSI off for deterministic eval
├── data/
│   ├── single_trajectory/        # one retargeted pkl per object (source for data_replay)
│   └── example_data/             # ready-to-train npz, one per object (output of data_replay)
├── scripts/
│   ├── data_replay.py            # pkl -> npz with sim-rolled body/object trajectories
│   ├── sample_object_points.py   # sample (P, 3) surface points from .obj per object
│   ├── list_envs.py
│   ├── zero_agent.py             # spawn env, step with zero action
│   ├── random_agent.py           # spawn env, step with random action
│   └── rsl_rl/
│       ├── train.py              # PPO training entry (defaults to headless)
│       ├── play.py               # checkpoint replay + JIT/ONNX export
│       └── cli_args.py           # RSL-RL specific argparse helpers
└── source/g1_hoi_learning/g1_hoi_learning/
    ├── algorithms/
    │   └── ppo/                  # per-algorithm subpackage: networks + algorithm + runner
    │       ├── networks.py       # SimBa actor-critic (LayerNorm + residual MLP)
    │       ├── algorithm.py      # MuonPPO (Muon + AdamW mixed optimizer wrapping PPO)
    │       └── runner.py         # re-export of RSL-RL OnPolicyRunner
    ├── assets/                   # G1 URDF + meshes
    ├── objects/                  # 13 household object configs + USD/OBJ + surface.npy
    ├── robots/g1_inspire.py      # G1 + Inspire hand articulation cfg
    └── tasks/manager_based/g1_hoi_learning/
        ├── __init__.py           # gym.register with _make_env factory (per-pkl object resolve)
        ├── g1_hoi_learning_env_cfg.py   # ManagerBasedRLEnvCfg (scene/obs/act/rew/term)
        ├── agents/rsl_rl_ppo_cfg.py      # PPORunnerCfg (SimBa + MuonPPO knobs)
        └── mdp/                          # commands, observations, rewards, terminations
```

Registered task: **`G1-Inspire-HOI-v0`** (single policy per training run; the npz `object_name` field selects which USD spawns).

---

## Installation

1. Install Isaac Lab via the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). This project assumes **Isaac Sim 5.1 + Isaac Lab 2.3.2 + Python 3.11 + PyTorch 2.11+**.

2. Clone this repo outside the `IsaacLab` directory.

3. Install the extension in editable mode using your Isaac Lab Python interpreter:

   ```bash
   python -m pip install -e source/g1_hoi_learning
   ```

4. Sideline Isaac Sim's bundled torch/nvidia (required for Muon optimizer + venv torch ABI):

   ```bash
   mv $ISAACSIM/exts/omni.isaac.ml_archive/pip_prebundle/torch  $ISAACSIM/exts/omni.isaac.ml_archive/pip_prebundle/torch.bak
   mv $ISAACSIM/exts/omni.isaac.ml_archive/pip_prebundle/nvidia $ISAACSIM/exts/omni.isaac.ml_archive/pip_prebundle/nvidia.bak
   ```

5. Verify:

   ```bash
   python scripts/list_envs.py
   ```

---

## Data Pipeline

### Step 1 — sample surface points (one-time, per object)

For each object, generate `surface.npy` (P=512 surface points in object-local frame, used by the `object_nearest_point_b` observation):

```bash
python scripts/sample_object_points.py --name clothesstand --num_points 512
# repeat for floorlamp, largebox, ...
```

Output: `source/g1_hoi_learning/g1_hoi_learning/objects/<name>/surface.npy`.

### Step 2 — pkl → npz (kinematic replay through Isaac Sim)

Take one retargeted trajectory pkl, kinematically replay it through the sim, capture per-frame body/object world states, save to npz:

```bash
python scripts/data_replay.py \
    --input_file ./data/single_trajectory/sub17_smallbox_001.pkl \
    --output_file ./data/example_data/smallbox.npz \
    --input_fps 30 --output_fps 50
```

The output npz contains: `joint_pos`, `joint_vel`, `body_pos_w/quat_w/lin_vel_w/ang_vel_w`, `object_pos_w/quat_w/lin_vel_w/ang_vel_w`, `contact_label`, `object_name`. The `object_name` field is read by `_make_env` at gym.make time to spawn the correct USD into the scene.

`data/example_data/` already contains pre-converted npz for 12 objects.

---

## Training

Training uses a **Hydra YAML config** (`configs/train.yaml`) that overrides the registered base task. Edit the YAML to change motion file, num_envs, reward weights, network size, etc.

### Default usage

```bash
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name train
```

`train.py` defaults to **headless mode** — no need to pass `--headless`.

### Inline overrides

Hydra-style `key=value` overrides still work alongside the YAML:

```bash
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name train \
    env.scene.num_envs=2048 \
    agent.max_iterations=500 \
    agent.algorithm.learning_rate=5.0e-4
```

### Resume from checkpoint

```bash
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name train \
    --resume --load_run 2026-05-09_14-23-11 --checkpoint model_500.pt
```

### Custom run name

```bash
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name train \
    --run_name smallbox-baseline
```

Logs go to `logs/rsl_rl/g1_inspire_hoi/<timestamp>_<run_name>/`.

### Available motion files

| object       | npz                                  |
| ------------ | ------------------------------------ |
| clothesstand | `data/example_data/clothesstand.npz` |
| floorlamp    | `data/example_data/floorlamp.npz`    |
| largebox     | `data/example_data/largebox.npz`     |
| largetable   | `data/example_data/largetable.npz`   |
| plasticbox   | `data/example_data/plasticbox.npz`   |
| smallbox     | `data/example_data/smallbox.npz`     |
| smalltable   | `data/example_data/smalltable.npz`   |
| suitcase     | `data/example_data/suitcase.npz`     |
| trashcan     | `data/example_data/trashcan.npz`     |
| tripod       | `data/example_data/tripod.npz`       |
| whitechair   | `data/example_data/whitechair.npz`   |
| woodchair    | `data/example_data/woodchair.npz`    |

### TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --port 6006
# open http://localhost:6006
```

---

## Evaluation / Play

Eval uses `configs/play.yaml`, which inherits from `train.yaml` and overrides
`num_envs=1` and turns off RSI / reset perturbations for deterministic playback.

```bash
# play the most recent checkpoint
python scripts/rsl_rl/play.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name play

# switch object inline (or edit play.yaml)
python scripts/rsl_rl/play.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name play \
    env.commands.motion.motion_file=./data/example_data/floorlamp.npz

# specific run / checkpoint
python scripts/rsl_rl/play.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name play \
    --load_run 2026-05-09_14-23-11 --checkpoint model_2000.pt

# record a video
python scripts/rsl_rl/play.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name play \
    --video --video_length 500
```

`play.py` automatically:
- runs in non-headless mode (GUI),
- exports `policy.pt` (TorchScript) and `policy.onnx` to `<run_dir>/exported/`.

---

## Sanity / Smoke Tests

Verify env construction and observation shapes without RL — reuses the play
config (single env, deterministic):

```bash
python scripts/zero_agent.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name play

python scripts/random_agent.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs --config-name play
```

---

## MDP Summary

| Component   | Detail |
| ----------- | ------ |
| Robot       | G1 (29 DOF body) + 2× Inspire hand (12 DOF/hand, follower joints driven by PhysX mimic gear constraints) |
| Object      | spawned per `motion_file["object_name"]` → `OBJECT_CFG_BY_NAME` lookup → matching USD |
| Action      | `JointPositionActionCfg` over driver joints (`^(?!.*(intermediate\|distal)).*$`) → 41 DoF |
| Observation | reference future motion (joint pos/vel + body pose + object pose + contact label, K=[0,1,2,4,8] frames), current robot state (body pose / base vel / joint pos/vel / last action), `object_nearest_point_b` (per-body to nearest surface point), live contact |
| Reward      | exp tracking on anchor pos/ori, body pos/ori/lin_vel/ang_vel, object pos/ori; per-body contact reward with continuous saturating-force (saturate_force=5 N); regularizers: `action_rate_l2 (-0.1)`, `joint_limit (-10)` |
| Termination | time_out (10 s) + anchor pos > 0.25 m / anchor ori > 0.8 rad / object pos > 0.25 m / object ori > 0.8 rad / EE z deviation > 0.25 m / hand-contact lost > 20 frames |
| Reset       | RSI (`rsi=True`): uniform random frame from motion + small pose / velocity / joint perturbation; eval (`rsi=False`): frame 0 |

### Network + algorithm

- **Backbone**: SimBa (input projection → N residual LN-MLP blocks → post-LN → output projection). Actor + critic each `hidden_dim=2048, num_blocks=2, expansion=1`. Both wrapped in `torch.compile(mode="default")`.
- **Algorithm**: `MuonPPO` (custom PPO subclass). 2D matrix params go through Muon optimizer with `match_rms_adamw` LR scaling; 1D params + scalars go through AdamW. Adaptive KL learning-rate schedule, GAE (γ=0.99, λ=0.95), 5 epochs × 4 mini-batches per iter.
- **Sim**: 4096 envs (configurable), `sim.dt=1/200 s`, `decimation=4` (policy at 50 Hz), episode 10 s, `solver_position_iter=4`, `enabled_self_collisions=True`, mimic gear constraints active.

---

## Code Formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

ruff (line length 120, py3.10+ target) + codespell.

---

## Design docs

- [docs/multi-object.md](docs/multi-object.md) — plan for training one policy across multiple objects (multi-asset scene + flat-concat motion buffer + `TensorClass`)
- [docs/psi.md](docs/psi.md) — plan for Physical State Initialization (replay-style reset state buffer, deferred)

---

## Troubleshooting

### Pylance missing indexing

Add to `.vscode/settings.json` under `"python.analysis.extraPaths"`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-this-repo>/source/g1_hoi_learning"
    ]
}
```

### Pylance out-of-memory

Comment out heavy omniverse packages:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*",
"<path-to-isaac-sim>/extscache/omni.kit.*",
"<path-to-isaac-sim>/extscache/omni.graph.*",
"<path-to-isaac-sim>/extscache/omni.services.*"
```

### Resolved scene.object printed but training crashes immediately

If you see `[INFO]: Resolved scene.object = ...` followed by a PhysX error, check the patch buffer:

```python
self.sim.physx.gpu_max_rigid_patch_count = 16 * 2**16   # bump if "Patch buffer overflow"
```
