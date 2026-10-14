# Depth Student-Weave / online action DAgger

The student replaces the teacher's entire `object_state` MLP branch with a
four-convolution depth encoder. The other three MLPs, their normalization
statistics, and SimBa are copied from the teacher run. Online training updates
**CNN and SimBa jointly**, while the vector MLPs, all normalization statistics,
and teacher remain fixed. Architecture is read from the teacher configuration.
No critic or reward objective is trained.

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
Missing absolute package asset paths are relocated only when the same relative
file exists in the local `g1_hoi_learning` package.

`--motion_files` is optional if the saved teacher motion paths remain valid.
Multiple training files/objects are supported. Training uses RSI to sample clips
within each object's environments. Keep validation clips separate; variants of
the same original motion should stay in the same split.

The YAML is **not** a Hydra config. Edit `training` in
[train.yaml](../../configs/dagger/train.yaml), or use `--num_envs` (also
`--num-envs`), `--seed`, `--device`, `--output_dir`, `--micro_batch_size`, and
`--max_iterations`. The latter limits this invocation without changing the
configured total iterations, for a short smoke run followed by resume.

## Network and observations

```text
ref_motion_body   -> fixed teacher-initialized MLP --+
ref_motion_object -> fixed teacher-initialized MLP --+
head_depth        -> trainable 4-layer CNN ----------+-> concat -> trainable SimBa -> raw action
robot_proprio     -> fixed teacher-initialized MLP --+
```

Depth is single-frame image-plane depth, rendered at 128x128 using the existing
camera mount and optics. Online observations and rollout storage use millimeter
quantization: invalid/out-of-range pixels become zero, valid pixels within
0.1-10 meters are divided by 10 meters. No per-image normalization or extra
mask channel is used. Input is `[B,1,128,128]`.

CNN channels are `1->32->64->128->128`; each convolution has kernel 3, stride 2,
padding 1, and SiLU. Final `[B,128,8,8]` features are flattened and projected to
the teacher latent dimension. The student reads only its three vector groups
and depth. Privileged object observations are used only by teacher labels.
Future object references and anchor-frame reference calculations are retained.

## Online training loop

This is a **single-process, single-GPU synchronous loop**:

1. Execute student mean actions for `rollout_steps`, with parameters fixed.
2. Label every pre-action state using the frozen teacher mean action.
3. Train only this rollout for `num_learning_epochs`.
4. Continue collecting with the updated student from the current observations.

A rollout is a short window, not a complete episode. Window boundaries do not
reset environments; ordinary episode termination/reset continues independently.
Environment stepping pauses during optimization. There is no teacher execution,
beta schedule, staged unfreezing, historical replay, or cross-window dataset
aggregation. This uses DAgger's labeling of student-visited states, with only the
current window retained rather than classical accumulated-dataset DAgger.

| Default setting | Value |
|---|---:|
| Environments | 4096 |
| Steps per window | 16 |
| Samples per window | 65,536 |
| Epochs per window | 2 |
| Minibatches per epoch | 32 |
| Effective samples per minibatch | 2,048 |
| Microbatch size | 128 |
| Backwards accumulated per optimizer update | 16 |
| Optimizer updates per window | 64 |
| Total iterations | 11,000 |

Every epoch independently shuffles and visits every sample exactly once.
If sample count is not divisible by minibatch count, balanced partitions retain
all samples. A microbatch decodes/transfers only its own observations and labels
to GPU. Its mean loss is weighted by `micro_size / actual_minibatch_size`,
including a shorter final microbatch. Gradients are zeroed before the minibatch;
clipping and AdamW step happen **once after all its microbatches**.
`micro_batch_size: null` processes the full effective minibatch.

Loss is raw-action MSE. Rewards and termination outcomes are logged but do not
enter the loss. The environment retains action clipping, scaling, offsets and
mimic joints; `last_action` reflects the action actually executed. AdamW learning
rates are 1e-4 for CNN and 1e-5 for SimBa, with gradient norm clipped to 1.0.

CPU storage is preallocated once and reused every window, avoiding a full
concatenation copy. Depth is uint16 (2 GiB at the default sample count), alongside
vectors, raw teacher actions and clip/episode/step metadata. Failed trajectories
are retained in the current window. CNN features are recomputed during training.

A randomly initialized CNN may yield frequent early failures under pure student
execution. An existing student can initialize a **fresh run**:

```bash
python scripts/dagger/train.py \
  --teacher_checkpoint /path/to/the/same/teacher/model_2000.pt \
  --student_checkpoint /path/to/student_weights.pt
```

Both v1 and v2 student snapshots are accepted when actor specification and teacher
identity match. Prefer lightweight weights over old full checkpoints containing
large replay arrays. SimBa and vector branches retain the loaded weights; CNN
and SimBa then train, while vector branches stay fixed. This restores no optimizer
or training counters. TorchScript `student_policy.pt` is an inference artifact,
not a warm-start checkpoint.

## Outputs and resume

Runs default to `logs/dagger/<timestamp>/` and contain:

- `params/env.yaml`, `params/dagger.yaml`: resolved environment and training contracts.
- `tensorboard/`, `metrics.jsonl`: rollout/train MSE, gradient norm, samples,
  optimizer update counts, rewards/resets, rollout/training times and peak memory.
- `model_<completed_iterations>.pt`: v2 student, AdamW, iteration/sample/update
  counters, RNG states and teacher identity; no rollout or historical replay.
- `student_weights.pt`: lightweight evaluation/warm-start weights at invocation end.
- `student_policy.pt`: student-only TorchScript export at invocation end.

Full checkpoints are saved every 100 iterations and at invocation end. Only the
newest `keep_last_checkpoints: 2` numbered full checkpoints in that run are
retained, pruned only after successfully committing the new checkpoint.

```bash
python scripts/dagger/train.py --resume logs/dagger/<run>/model_100.pt
tensorboard --logdir logs/dagger
```

Resume uses the saved training configuration/environment and checks teacher SHA256
and actor/action/observation contracts. Physics is not serialized: resume starts
new episodes while retaining learning state. Snapshots occur only at completed
rollout/update boundaries. Restart interrupted partial iterations from the last
complete snapshot. `--micro_batch_size` may change on resume without changing
effective minibatches or update counts. Old v1 staged/replay checkpoints cannot
resume this schedule; use `--student_checkpoint` for a fresh run instead.

Peak RAM and PyTorch allocated/reserved CUDA memory are logged. Renderer
allocations are outside PyTorch counters, so inspect `nvidia-smi` as well.

## Evaluate

Use held-out clips for **one object per invocation**, following Weave's current
`eval_mode` limitation. Evaluation automatically records an MP4 for every clip
by default, using a single environment and the same rollout as the reported
metrics. The scene is created once and reused across clips. Student depth inputs
remain 128x128; the video uses an external RGB camera framing the robot and
object together, increasing viewing distance when the object is farther away.
Video mode uses one environment even if `--num_envs 32` is passed, and prints
the actual configuration at startup.

Set `video.enabled: false` in [eval.yaml](../../configs/dagger/eval.yaml), or pass
`--no_video`, for the original parallel evaluation (default batch limit 32).
The final batch's unused slots repeat a valid clip and are excluded from the report.

```bash
python scripts/dagger/eval.py \
  --checkpoint logs/dagger/<run>/model_11000.pt \
  --motion_files /path/to/held_out_floorlamp.npz

python scripts/dagger/eval.py \
  --checkpoint logs/dagger/<run>/model_11000.pt \
  --motion_files /path/to/held_out_floorlamp.npz \
  --policy teacher

# Faster parallel metrics without recording:
python scripts/dagger/eval.py \
  --checkpoint logs/dagger/<run>/student_weights.pt \
  --motion_files /path/to/held_out_floorlamp.npz \
  --no_video --num_envs 32
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

Videos are saved as `eval_<policy>/videos/clip_<id>_<motion_name>.mp4`, with
sanitized names and clip ids to distinguish duplicate motion names. The default
is H.264 RGB video at 1280x720, sampled every two control steps: 25 FPS at the
current 50 Hz control rate. `video.width`, `video.height`, and `video.frame_stride`
are configurable; dimensions must be positive and even. Frames are streamed
through ImageIO/FFmpeg rather than buffered for the whole episode. The runtime
needs `imageio` and `imageio-ffmpeg` (both are present in the current simulator environment).

Each clip records its initial frame and terminal frame; the latter is captured
by an Isaac Lab post-step recorder callback **before automatic environment reset**.
Successful and failed episodes both produce videos. The recorder never appends
the subsequent reset state. Each clip row includes `video_path` (relative to the
report directory), `video_frames`, `video_fps`, `video_complete`,
`video_frame_steps`, and `video_simulation_seconds`. An interrupted rollout is
marked incomplete, and a playable partial recording is finalized when possible.
The extra initial/terminal frames may extend playback by up to two video frames.

`metrics.json` is atomically updated after every clip/batch. MP4s are written as
`.partial.mp4` until the encoder closes successfully, then renamed to their final
names. Re-running evaluation into the same output directory replaces matching
clip videos and the report; use `--output_dir` to retain separate evaluations.

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
