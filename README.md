# G1 HOI Learning

Isaac Lab extension for **Unitree G1 + Inspire dexterous hands** human-object-interaction (HOI) motion imitation. 
A single PPO policy learns to mimic mocap-retargeted reference motions of the robot manipulating one or more household objects (tripod, suitcase, chairs, boxes, ...) while satisfying contact constraints on the hands. 
Parallel envs are assigned objects round-robin and each env samples among that object's reference clips, so one run trains across many objects and thousands of clips.

Built on top of Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + RSL-RL.

---

## Project Layout

```
g1_hoi_learning/
├── configs/
│   ├── train.yaml                # Hydra config: env / agent overrides for training
│   └── play.yaml                 # inherits train.yaml; num_envs=1, RSI off for deterministic eval
├── data/
│   ├── retargeted/               # <obj>_train.pkl — per-object dict of retargeted clips
│   │                             #   (merged + no-hand-contact clips filtered out; input to packing)
│   ├── train/                    # <obj>_<N>clip.npz — packed multi-clip TRAIN sets (one per object)
│   ├── test/                     # <obj>_<N>clip.npz — packed multi-clip TEST sets (one per object)
│   ├── example_data/             # single-clip example npz, one per object (quick smoke runs)
│   └── plot/                     # dataset-distribution figure + script (train/test composition)
├── scripts/
│   ├── data_replay_multiple.py   # pack ALL clips of one object (a dict pkl) → one multi-clip npz
│   ├── data_replay.py            # single trajectory pkl → npz (one clip)
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
    ├── objects/                  # household object configs + USD/OBJ + surface.npy
    ├── robots/g1_inspire.py      # G1 + Inspire hand articulation cfg
    └── tasks/manager_based/g1_hoi_learning/
        ├── __init__.py           # gym.register with _make_env factory (per-npz object resolve)
        ├── g1_hoi_learning_env_cfg.py   # ManagerBasedRLEnvCfg (scene/obs/act/rew/term)
        ├── agents/rsl_rl_ppo_cfg.py      # PPORunnerCfg (SimBa + MuonPPO knobs)
        └── mdp/                          # commands, observations, actions, rewards, terminations
```

Registered task: **`G1-Inspire-HOI-v0`** (single policy per training run). 
Each motion npz carries an `object_names` field; listing several npz in the config trains one policy across all of them.
Environments are assigned objects round-robin and resolve the matching USD at `gym.make` time.

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

### Step 2 — pack all clips of one object → one multi-clip npz

`data_replay_multiple.py` kinematically replays every clip of an object through Isaac Sim, captures per-frame body/object world states, and concatenates them into a single packed npz for multi-clip RL training:

```bash
python scripts/data_replay_multiple.py \
    --input_file  ./data/retargeted/smalltable_train.pkl \
    --output_file ./data/train/smalltable_279clip.npz \
    --input_fps 30 --output_fps 50 --headless --overwrite
```

The packed npz contains the per-frame arrays `joint_pos`, `joint_vel`, `body_pos_w/quat_w/lin_vel_w/ang_vel_w`, `object_pos_w/quat_w/lin_vel_w/ang_vel_w`, `contact_label` (all clips concatenated along axis 0), plus `motion_lengths` (frames per clip), `object_names`, `motion_names`, and `fps`. The `MotionLoader` slices clips back out via the `motion_lengths` offsets and groups them by object.

---

## Training

Training uses a **Hydra YAML config** (`configs/train.yaml`) that overrides the registered base task. Edit the YAML to change motion files, num_envs, reward weights, network size, etc. The default config trains across all 11 objects in `data/train/`.

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
    env.scene.num_envs=8192 \
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
    --run_name multiobj-baseline
```

Logs go to `logs/rsl_rl/g1_inspire_hoi/<timestamp>_<run_name>/`.

### Dataset (motion files)

`configs/train.yaml` lists all 11 train sets; `configs/play.yaml` points at the matching test sets. Clip counts (no-hand-contact clips already filtered out):

| object       | train npz                       | test npz                      |
| ------------ | ------------------------------- | ----------------------------- |
| clothesstand | `train/clothesstand_311clip.npz`| `test/clothesstand_31clip.npz`|
| floorlamp    | `train/floorlamp_289clip.npz`   | `test/floorlamp_34clip.npz`   |
| largebox     | `train/largebox_287clip.npz`    | `test/largebox_48clip.npz`    |
| largetable   | `train/largetable_270clip.npz`  | `test/largetable_36clip.npz`  |
| smallbox     | `train/smallbox_252clip.npz`    | `test/smallbox_38clip.npz`    |
| smalltable   | `train/smalltable_279clip.npz`  | `test/smalltable_42clip.npz`  |
| suitcase     | `train/suitcase_290clip.npz`    | `test/suitcase_25clip.npz`    |
| trashcan     | `train/trashcan_253clip.npz`    | `test/trashcan_35clip.npz`    |
| tripod       | `train/tripod_429clip.npz`      | `test/tripod_34clip.npz`      |
| whitechair   | `train/whitechair_393clip.npz`  | `test/whitechair_38clip.npz`  |
| woodchair    | `train/woodchair_383clip.npz`   | `test/woodchair_54clip.npz`   |
| **total**    | **3,436 clips (~5.9 h)**        | **415 clips (~38 min)**       |

A figure of the train/test composition by object is in `data/plot/` (regenerate with `python data/plot/plot_distribution_stacked_bar.py`).

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
    env.commands.motion.motion_files=[./data/test/floorlamp_34clip.npz]

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

> Note: each episode ends at `episode_length_s` (10 s = 500 steps) or a divergence
> termination, independent of clip boundaries; the command walks through clips on
> reset. To evaluate exactly one clip per episode you must terminate on clip end.

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
| Robot       | G1 (29 DOF body) + 2× Inspire hand (12 DOF/hand: 6 driver + 6 follower joints; followers driven in software from a `mimic` table) |
| Object      | `_make_env` reads each motion npz's `object_names`, looks it up in `OBJECT_CFG_BY_NAME`, and builds a round-robin `MultiAssetSpawnerCfg` (env `i` → object of `motion_files[i % N]`) |
| Action      | `MimicJointPositionActionCfg` over driver joints (`^(?!.*(intermediate\|distal)).*$`) → 41 DoF; follower finger joints written each step as `driver * mult + offset` |
| Observation | reference future motion (joint pos/vel + body pose + object pose + contact label, K=[0,1,2,4,8] frames), current robot state (body pose / base vel / joint pos/vel / last action), `object_nearest_point_b` (per-body to nearest surface point), live contact |
| Reward      | exp tracking on anchor pos/ori, body pos/ori, object pos/ori, and hand-object relative position in the object frame (std=0.1, weight 2.0); per-body contact reward with continuous saturating-force (saturate_force=5 N); regularizers: `action_rate_l2 (-0.1)`, `joint_limit (-10)` |
| Termination | time_out (10 s) + anchor pos > 0.25 m / anchor ori > 0.8 rad / object pos > 0.25 m / object ori > 0.3 rad / EE z deviation > 0.25 m / hand-contact lost > 20 frames |
| Reset       | RSI (`rsi=True`): random clip of the env's object + uniform random frame + small pose / velocity / joint perturbation; eval (`rsi=False`): walk clips from frame 0 |

### Network + algorithm

- **Backbone**: SimBa (input projection → N residual pre-LN MLP blocks → post-LN → output projection). Actor + critic each `hidden_dim=2048, num_blocks=2, expansion=1`.
- **Algorithm**: `MuonPPO` (custom PPO subclass). 2D matrix params go through Muon optimizer with `match_rms_adamw` LR scaling; 1D params + scalars go through AdamW. Adaptive KL learning-rate schedule, GAE (γ=0.99, λ=0.95), 5 epochs × 4 mini-batches per iter.
- **Sim**: 4096 envs (configurable), `sim.dt=1/200 s`, `decimation=4` (policy at 50 Hz), episode 10 s, `solver_position_iteration_count=8`, `enabled_self_collisions=True`. Inspire-hand followers use software mimic — no PhysX gear constraint.

---

## Code Formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

ruff (line length 120, py3.10+ target) + codespell.