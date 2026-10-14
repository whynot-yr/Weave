# Depth Student-Weave / action DAgger

The teacher PPO task and runner are unchanged. The student replaces the entire
`object_state` MLP branch with a four-convolution depth encoder. The other three
MLPs, their normalization statistics, and SimBa are copied from the teacher run.
The actor architecture is read from that run's configuration, not guessed from
the current PPO defaults. No critic is trained during DAgger.

## Train

Run from the Weave repository with the Isaac Sim environment activated:

```bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/dagger/train.py \
  --config configs/dagger/train.yaml \
  --teacher_checkpoint /path/to/teacher/run/model_2000.pt \
  --motion_files /path/to/training_motions.npz
```

`params/agent.yaml` and `params/env.yaml` are discovered next to the teacher
checkpoint. If absent, supply `--teacher_agent_config` and `--teacher_env_config`
with the exact configurations used for that checkpoint. Both IsaacLab snapshots
and mappings with an `agent`/`env` root are accepted. Actor weights load strictly.
The environment factory rebuilds the object asset from the motion files.
Missing absolute paths to package assets from another machine are relocated
only when the same relative file exists in the local `g1_hoi_learning` package.

`--motion_files` is optional if the saved teacher motion paths are still valid.
Multiple training files/objects are supported. Training uses RSI so clips are
resampled within each object's environments. Keep validation clips separate;
variants of the same original motion should stay in the same split.

The independent YAML is **not** a Hydra config. Edit `training` in
[train.yaml](../../configs/dagger/train.yaml), or use the supported CLI options
`--num_envs`, `--seed`, `--device`, `--output_dir`, `--micro_batch_size`, and `--max_iterations`.
The latter limits this invocation without changing the stage schedule, which is
useful for a short smoke run followed by resume.

## Network and observations

```text
ref_motion_body   -> teacher-initialized MLP --+
ref_motion_object -> teacher-initialized MLP --+
head_depth        -> 4-layer CNN --------------+-> concat -> teacher-initialized SimBa -> raw action
robot_proprio     -> teacher-initialized MLP --+
```

Depth is single-frame image-plane depth, directly rendered at 128x128 using the
existing camera mount and optics. Online observations and replay both use
millimeter quantization: invalid/out-of-range pixels become zero, valid pixels
within 0.1–10 meters are divided by 10 meters. No per-image normalization or
extra mask channel is used. Input is `[B,1,128,128]`.

CNN channels are `1->32->64->128->128`; each convolution has kernel 3, stride 2,
padding 1, and SiLU. The final `[B,128,8,8]` features are flattened and projected
to the teacher's latent dimension. Vector normalization stays fixed throughout.

The environment still computes privileged object observations for teacher
labels. The student forward only reads its three vector groups and depth.
The policy remains reference-conditioned: future object references and the
anchor-frame reference calculations are retained.

## Training stages

| Stage | Default iterations | Execution | Updated student parameters |
|---|---:|---|---|
| Visual BC warmup | 500 | Teacher mean action | CNN only |
| Joint BC warmup | 500 | Teacher mean action | CNN, three MLPs, SimBa |
| DAgger handover | 5000 | Per-env/per-step teacher with probability beta | All actor parameters |
| Pure-student DAgger | 5000 | Student mean action | All actor parameters |

Beta decreases linearly from 1 towards 0 during handover, then becomes exactly
0 in the final stage. Teacher and student actions are deterministic means.
Every state is labeled by the frozen teacher **before** stepping the environment,
regardless of which action is executed. Loss is raw-action MSE; rewards and
termination outcomes are logged but do not enter the loss. The original action
clipping, scaling, offsets, and mimic joints are applied by the environment.
`last_action` is always the actual executed action.

Each iteration collects 32 steps per environment and performs 20 AdamW updates
with effective batch size 256 across 32 environments (1,024 new samples).
`micro_batch_size: null` processes the full batch. On a GPU with limited memory,
set it to 64 (or pass `--micro_batch_size 64`) to accumulate four backwards
before each optimizer update. Losses are weighted by sample count, including
an uneven final microbatch, and gradient clipping happens only after accumulation.
This retains 20 optimizer steps per iteration, rather than multiplying them by four.
Default learning rates are 1e-4 for CNN and 1e-5 for the
transferred MLP/SimBa parameters; gradient norm is clipped to 1.0. Unlocking
the transferred parameters preserves the CNN's AdamW state.

Minibatches use 50% current-rollout samples and 50% historical samples; the first
iteration uses only current samples. After updating, current samples enter a
10,000-sample reservoir. Failed trajectories are retained. CPU replay stores
depth as uint16 (312.5 MiB for depth alone at default capacity), vectors and
raw teacher actions, plus clip/episode/step/execution metadata. CNN latents are
recomputed at every update.

## Outputs and resume

Runs default to `logs/dagger/<timestamp>/` and contain:

- `params/env.yaml`, `params/dagger.yaml`: resolved environment and training contracts.
- `tensorboard/`, `metrics.jsonl`: MSE, beta, execution fractions, rewards, resets, and replay size.
- `model_<completed_iterations>.pt`: student, AdamW, stage counters, replay, RNG states, and teacher identity.
- `student_<stage>.pt`: lightweight student weights and evaluation metadata at each completed stage; no replay or optimizer.
- `student_policy.pt`: student-only TorchScript export after a completed invocation.

Full checkpoints are saved every 100 iterations by default and at invocation end.
Only the newest `keep_last_checkpoints: 2` numbered full checkpoints are retained
in that run's directory. Older full snapshots are removed only after the new
checkpoint is successfully committed. Lightweight stage weights are retained
and can be passed to `scripts/dagger/eval.py --checkpoint`; they cannot resume training.
No separate archive of all collected depth images is written.

```bash
python scripts/dagger/train.py --resume logs/dagger/<run>/model_100.pt
tensorboard --logdir logs/dagger
```

Resume restores the saved training schedule and environment, and checks the
teacher SHA256, actor, action, and observation contracts. Replay is included
in each checkpoint, so full checkpoints are large. Simulator physics is not
serialized: resume starts new episodes while preserving the learning state.
Snapshots are written only at completed rollout/update boundaries. Interrupted
partial iterations should be restarted from the last complete snapshot.
The memory-only `--micro_batch_size` override can be changed on resume without
changing effective batch size, stage durations, or optimizer step counts.
Metrics include process peak RAM and PyTorch peak allocated/reserved CUDA memory;
use `nvidia-smi` as well because renderer allocations are outside PyTorch's counters.

## Evaluate

Use held-out clips for **one object per invocation**, following Weave's current
`eval_mode` limitation. Evaluation batches camera environments (default 32),
so every clip can be evaluated without rendering all clips simultaneously.
The scene is created once and reset onto successive clip batches; the last
batch's unused slots repeat a valid clip and are excluded from the report.

```bash
python scripts/dagger/eval.py \
  --checkpoint logs/dagger/<run>/model_11000.pt \
  --motion_files /path/to/held_out_floorlamp.npz

python scripts/dagger/eval.py \
  --checkpoint logs/dagger/<run>/model_11000.pt \
  --motion_files /path/to/held_out_floorlamp.npz \
  --policy teacher
```

Both use deterministic evaluation, no domain randomization, no proprio noise,
and identical clips/terminations. Student evaluation never uses teacher actions
or updates parameters. Teacher baseline identity is checked against training;
`--teacher_checkpoint` supports relocating that same file.

Reports go to `eval_student/metrics.json` and `eval_teacher/metrics.json`, or
`--output_dir`. They include original command tracking metrics, per-clip
success/progress/failure reasons, missed reference contact bodies per step,
action-change MSE, and synchronized batch inference latency. Camera rendering
and environment stepping are outside the reported policy inference latency.

## Export interface and checks

The exported student accepts four tensors in this order:

```text
ref_motion_body:   [B, D_body]    (raw vector, normalization included)
ref_motion_object: [B, D_object]  (raw vector, normalization included)
robot_proprio:    [B, D_proprio] (raw vector, normalization included)
depth_m:          [B, 1, 128, 128] (image-plane depth in meters, preprocessing included)
```

It returns `[B, num_actions]` raw actions. Action processing remains external.

Simulator-free contract tests:

```bash
/data/g1_hoi_ws/sim51/bin/python -m pytest source/g1_hoi_learning/tests/dagger -q
```
