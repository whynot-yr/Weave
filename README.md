# g1_hoi_learning

Isaac Lab extension for **Unitree G1 + Inspire dexterous hands** human-object-interaction (HOI) motion imitation. A single PPO policy learns to mimic mocap-retargeted reference motions of the robot manipulating a household object (clothesstand, suitcase, monitor, ...) while satisfying contact constraints on the hands.

Built on top of Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + RSL-RL.

---

## Project Layout

```
g1_hoi_learning/
├── data/
│   ├── single_trajectory/       # extracted single-trajectory pkl for replay/training
│   └── example_data/            # ready-to-train npz (one file per object)
├── docs/
├── scripts/
│   ├── data_replay.py           # pkl -> npz with sim-rolled body/object trajectories
│   ├── sample_object_points.py  # sample surface points from .obj for obs/reward
│   ├── list_envs.py
│   ├── zero_agent.py / random_agent.py
│   └── rsl_rl/{train,play}.py   # PPO entry points
└── source/g1_hoi_learning/g1_hoi_learning/
    ├── algorithms/muon_ppo.py   # Muon + AdamW mixed optimizer wrapping PPO
    ├── networks/simba.py        # SimBa actor-critic backbone (LayerNorm + residual MLP)
    ├── objects/                 # 13 household objects (USD + obj + sampled surface)
    ├── robots/g1_inspire.py     # G1 + Inspire hand articulation cfg
    └── tasks/manager_based/g1_hoi_learning/
        ├── g1_hoi_learning_env_cfg.py    # ManagerBasedRLEnvCfg (scene/obs/act/rew/term)
        └── mdp/                          # commands, observations, rewards, terminations
```

Registered task: **`G1-Inspire-HOI-v0`** (single policy per training run, motion file selects which object).

---

## Installation

1. Install Isaac Lab via the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). This project assumes Isaac Sim 5.1 + Isaac Lab 2.3.2 + Python 3.11 + PyTorch 2.11+.

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

For each object, generate `surface.npy` (uniformly sampled points on the object mesh, used for `object_nearest_point_b` obs):

```bash
python scripts/sample_object_points.py --name clothesstand --num_points 512
# repeat for floorlamp, largebox, ... or loop in a shell
```

### Step 2 — pkl → npz (kinematic replay through Isaac Sim)

Take a single retargeted trajectory pkl, replay it kinematically through the sim to capture per-frame body/object world states, save to npz:

```bash
python scripts/data_replay.py \
    --input_file ./data/single_trajectory/sub17_smallbox_001_retargeted.pkl \
    --output_file ./data/example_data/smallbox.npz \
    --input_fps 30 --output_fps 50
```

The output npz includes `joint_pos`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `object_pos_w/quat_w/lin_vel_w/ang_vel_w`, `contact_label`, `object_name`. The `object_name` field is used at training time to spawn the correct USD into the scene.

`./data/example_data/` already contains pre-converted npz for all 12 non-clothesstand objects.

---

## Training

`motion_file` is required (no default) — pass via Hydra override:

```bash
# train on smallbox (with 4096 envs, headless):
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 4096 --headless --max_iterations 2000

# train on a different object — just change the npz:
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/floorlamp.npz \
    --num_envs 4096 --headless --max_iterations 2000

# resume from checkpoint:
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 4096 --headless \
    --resume --load_run 2026-05-07_12-34-56 --checkpoint model_500.pt

# named run (output dir will use this suffix):
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 4096 --headless --run_name smallbox-baseline
```

Logs are written to `logs/rsl_rl/g1_inspire_hoi/<timestamp_runname>/`.

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

### Tensorboard

```bash
tensorboard --logdir logs/rsl_rl --port 6006
# open http://localhost:6006
```

> If TensorBoard reports "could not bind to port" while `lsof` shows nothing, you may have a `nft` redirect rule from clash-verge intercepting localhost. Add `tcp dport 6006 accept` to the `eset_eea_wap` output chain to whitelist that port.

---

## Evaluation / Play

Replay the most recent checkpoint (RSI off, starts from frame 0):

```bash
python scripts/rsl_rl/play.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 1
```

Specific run / checkpoint:

```bash
python scripts/rsl_rl/play.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 1 \
    --load_run 2026-05-07_12-34-56 --checkpoint model_2000.pt
```

`play.py` automatically:
- disables RSI (eval starts from frame 0)
- exports `policy.pt` (JIT) and `policy.onnx` to `<run_dir>/exported/`

Record a video of the rollout:

```bash
python scripts/rsl_rl/play.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 1 --video --video_length 500
```

---

## Sanity / Smoke Tests

Verify env construction and observation shapes without RL:

```bash
# zero-action agent (just spawn + step with action=0)
python scripts/zero_agent.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 4

# random-action agent
python scripts/random_agent.py --task=G1-Inspire-HOI-v0 \
    commands.motion.motion_file=./data/example_data/smallbox.npz \
    --num_envs 4
```

---

## MDP Summary

| Component   | Detail                                                                                                                                                                                                                                  |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Robot       | G1 (29 DOF) + 2× Inspire hand (12 DOF/hand, follower joints driven by PhysX mimic gear constraints)                                                                                                                                     |
| Object      | spawned per `motion_file["object_name"]` lookup into `OBJECT_CFG_BY_NAME` (13 objects available)                                                                                                                                        |
| Action      | `JointPositionActionCfg` over driver joints (`^(?!.*(intermediate\|distal)).*$`) → 41 DoF                                                                                                                                              |
| Observation | reference future motion (joint pos/vel @ K=[0,1,2,4,8] frames + body pose + object pose + contact label), current robot state (body pose / base vel / joint pos/vel / last action), `object_nearest_point_b` (per-body to surface), live contact |
| Reward      | exp tracking on anchor pos/ori, body pos/ori/lin_vel/ang_vel, object pos/ori; contact reward (per-body match with continuous saturating force, threshold 5 N); regularizers: `action_rate_l2`, `joint_limit`                                  |
| Termination | time_out (10 s) + bad anchor pos (>0.25 m) / ori (>0.8 rad) / object pos / object ori / EE z deviation / hand-contact lost > 20 frames                                                                                                  |
| Reset       | RSI (`rsi=True`): uniform random frame from motion + small pose / velocity / joint perturbation; eval (`rsi=False`): frame 0                                                                                                            |

### Network + algo

- **Backbone**: SimBa (input projection → N residual LN-MLP blocks → post-LN → output projection); actor + critic both `hidden_dim=2048, num_blocks=2, expansion=1`. `torch.compile(mode="default")` wrapping the two backbones.
- **Algorithm**: `MuonPPO` (custom PPO subclass). 2D matrix params go through Muon optimizer with `match_rms_adamw` LR scaling; 1D params + scalar params go through AdamW. Adaptive KL learning-rate schedule, GAE(γ=0.99, λ=0.95), 5 epochs × 4 mini-batches per iteration.
- **Sim**: 4096 envs, `sim.dt=1/200 s`, `decimation=4` (policy at 50 Hz), episode 10 s, `solver_position_iter=8`, `enabled_self_collisions=True`, mimic gear constraints active.

---

## Code Formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

ruff (line length 120, py3.10+) + codespell.

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

### Pylance crash (out of memory)

Comment out heavy omniverse packages:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*",
"<path-to-isaac-sim>/extscache/omni.kit.*",
"<path-to-isaac-sim>/extscache/omni.graph.*",
"<path-to-isaac-sim>/extscache/omni.services.*",
```

### TensorBoard "port already in use" but `ss` shows nothing

Likely a system-level proxy redirect (e.g. clash-verge `nft` rule) intercepting localhost. Whitelist your tb port in the redirect chain:

```bash
sudo nft insert rule ip eset_eea_wap output tcp dport 6006 accept
```

### `MissingMandatoryValue: Missing mandatory value: commands.motion.motion_file`

You forgot to pass the motion file. It's intentionally required — pick one from `data/example_data/`:

```bash
... commands.motion.motion_file=./data/example_data/smallbox.npz
```
