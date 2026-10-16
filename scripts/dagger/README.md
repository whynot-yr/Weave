# Current depth student + historical 3904770 PPO/DAgger

Restored from commit 3904770889ee67b34113df01f3f5533add0a468b under
algorithms/distillation/, adapted for the current student and depth storage.

~~~text
BC  = MSE(student action mean, frozen teacher action)
PPO = clipped policy loss + value_coef * clipped value loss - entropy_coef * entropy
loss = alpha * BC + (1 - alpha) * PPO
alpha = 0.5 * (1 + cos(pi * completed_iterations / 6000)), then 0 after 6000
~~~

Each iteration holds alpha fixed for all minibatches and advances it once after
all updates. First iteration is BC only; subsequent iterations gradually
increase PPO's contribution. After 6000 iterations training is pure PPO.
The teacher never controls the robot.

## Initialization and preserved perception

Fresh runs initialize from the **teacher**, not the previous model_1000:

- Student vector encoders and SimBa: copied from teacher Actor.
- Student vector normalizers: copied by group from teacher statistics and fixed.
- Current depth CNN: randomly initialized.
- Independent Critic and its normalizer: copied from teacher Critic.
- State-independent per-action Gaussian standard deviation: starts at 0.2.
- Optimizers, counters and BC schedule: start fresh.

CNN, vector encoders, SimBa, Critic and std are trainable. Teacher parameters and
**all normalization statistics stay fixed**. Unlike the historical environment,
this adapter also fixes the Critic normalizer.

~~~text
ref_motion_body   -> teacher-initialized, trainable MLP --+
ref_motion_object -> teacher-initialized, trainable MLP --+
head_depth        -> current trainable 4-layer CNN ------+-> concat -> SimBa -> action mean
robot_proprio     -> teacher-initialized, trainable MLP --+
~~~

Depth remains single-frame [B,1,128,128] from the existing TiledCamera.
Valid 0.1-10m depths are quantized to millimeters and divided by 10m; invalid
pixels become zero. CNN channels remain 1->32->64->128->128, with stride-two
3x3 convolutions, SiLU, and the existing latent projection. No extra image
normalization, noisy reference input, historical camera or depth degradation is
introduced. Teacher-provided motion/environment/reward definitions stay in effect.

Critic uses the teacher's critic groups (normally ref_motion_body,
ref_motion_object, object_state, robot_privileged). Privileged state is used
only by Critic and teacher. Student Actor does not read object_state or
robot_privileged. Critic does not share the CNN.

## Train

Run from the repository with the Isaac Sim environment activated:

~~~bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/dagger/train.py \
  --config configs/dagger/train.yaml \
  --teacher_checkpoint /path/to/teacher/run/model_31400.pt \
  --motion_files /path/to/training_motions.npz
~~~

Teacher params/agent.yaml and params/env.yaml are discovered next to the
checkpoint. If needed pass --teacher_agent_config and --teacher_env_config.
Actor, Critic and normalization weights load strictly. Without --motion_files,
saved teacher paths are used. Multiple objects are supported; RSI samples clips.

The standalone YAML is [train.yaml](../../configs/dagger/train.yaml), not Hydra.
CLI options include --num_envs/--num-envs, --micro_batch_size, --seed, --device,
--output_dir and --max_iterations. The last limits this invocation without
changing the 11000-iteration budget or 6000-iteration BC schedule.

## Two GPUs

~~~bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dagger/train_ddp.sh \
  --teacher_checkpoint /path/to/teacher/run/model_31400.pt \
  --motion_files /path/to/training_motions.npz
~~~

Each rank owns a simulator, camera, Actor/Critic, frozen teacher, optimizer and
current rollout. LOCAL_RANK binds physics/rendering before SimulationApp starts.
Environment, action-sampling and minibatch streams use seed + rank; initial
weights are synchronized.

| Default | Per rank | Two ranks |
|---|---:|---:|
| Environments | 4096 | 8192 |
| Rollout steps | 32 | 32 |
| Samples per rollout | 131072 | 262144 |
| Effective minibatch | 32768 | 65536 |
| Microbatch | 128 | 128 per rank |
| Microbatches per update | 256 | 256 per rank |
| Epochs | 5 | 5 |
| Optimizer updates per iteration | 20 | **20** |

Both GPUs collect, then both train. Stepping pauses during updates.
No historical replay is kept. Each epoch reshuffles and visits every local
sample; balanced partitions retain every sample. Only a microbatch is decoded
and moved to GPU. Loss weighting is microbatch size / effective minibatch size.
Non-final forward/backward calls use DDP no_sync; final backward synchronizes
accumulated Actor, Critic and std gradients. Clip/step run once per effective
minibatch. No additional historical manual gradient reduction is called.

RSL-RL PPO supplies action/value/log-prob recording, timeout reward bootstrap
and GAE. The storage adapter owns state BEFORE env.step, since simulator buffers
can change in place. RSL-RL GAE runs on CPU; advantage statistics are combined
across ranks. Teacher labels use the stored same-state observations.
Preallocated uint16 depth storage is about 4 GiB per rank at defaults and reused.

Existing Muon + AdamW starts at 1e-4, decay 0.01 (AdamW parameters), gradient norm
1.0. Global KL adapts learning rate with historical thresholds and synchronizes
it across ranks. Defaults: gamma 0.99, GAE lambda 0.95, clip 0.2,
value coefficient 1, entropy coefficient 0.0005.

## Outputs and resume

Runs default to logs/dagger/<timestamp>/:

- params/env.yaml, params/dagger.yaml: environment, teacher, Actor/Critic,
  training settings and source commit.
- tensorboard/, metrics.jsonl: BC, surrogate/value losses, entropy, KL,
  clip fraction, explained variance, std, LR, BC/PPO weights, rewards/resets,
  samples/updates, per-rank timing and memory.
- model_<completed_iterations>.pt: v3 training checkpoint.
- student_weights.pt: mean-Actor-only weights.
- student_policy.pt: mean-Actor-only TorchScript export.

**Every 100 completed iterations, save and keep all checkpoints by default:**
model_100.pt, model_200.pt, ..., model_11000.pt. Also save at invocation end.
Optional keep_last_checkpoints: N prunes only after a successful atomic save.

v3 saves Actor, Critic, fixed normalizers, std, both optimizer states, LR,
BC schedule, counters, teacher identity, environment, world size and each
rank's RNG. Only rank 0 writes. Physics is not serialized: resume starts
new episodes while restoring learning state.

~~~bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dagger/train_ddp.sh \
  --resume logs/dagger/<run>/model_100.pt
~~~

Strict resume requires v3, matching world size, per-rank environments and learning
configuration. Memory/save controls may change; current YAML controls save
interval/retention. Old v1/v2 models remain evaluable but cannot resume PPO.
Optional --student_checkpoint supports explicitly loading compatible old Actor
weights into a fresh run; **it is not used by default**.

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
and identical clips/terminations by default. Student evaluation never uses teacher actions
or updates parameters. Teacher baseline identity is checked against training;
`--teacher_checkpoint` supports relocating that same file.

To evaluate one clip repeatedly with sampled initial object poses:

```bash
python scripts/dagger/eval.py \
  --checkpoint outputs/dagger_ppo_floorlamp_033_run1/model_100.pt \
  --policy student \
  --motion_files data/test/floorlamp_033_var5.npz \
  --clip_ids 0 --num_trials 20 --perturbation initial_small --seed 42 \
  --device cuda:0 --num_envs 4 --no_video \
  --output_dir outputs/dagger_ppo_floorlamp_033_run1/100/eval_clip0_small
```

`--clip_ids` accepts zero-based ids across the supplied motion files; omit it to
evaluate every clip. `--num_trials` defaults to 1. With `--no_video`, environments
can run independent trials of the same clip in parallel. Omit `--no_video` to
record every trial serially; repeated trials have `_trial_<id>` video suffixes.
The same options can be set as `clip_ids`, `num_trials` and `perturbation` in YAML.

The initial-pose profiles sample independent uniform offsets per trial:

| `--perturbation` | World x and y, each | Yaw around world z |
| --- | --- | --- |
| `none` (default) | 0 | 0 |
| `initial_small` | ±0.002 m | ±0.5 degrees |
| `initial_medium` | ±0.005 m | ±1 degree |
| `initial_large` | ±0.010 m | ±2 degrees |

Only the object's actual initial pose changes, before observations and camera
frames are read. Rotation is about the object's root position. Object height,
world velocities, robot initial state and reference trajectories remain unchanged;
each trial starts at reference frame zero. Proprio noise, physical-parameter
randomization and pushes remain disabled. Check a few videos to choose ranges
appropriate to the clip's initial grasp and support geometry.

Each trial's sampling seed is derived from `(seed, clip_id, trial_id)`. Changing
batch size, clip selection order or the number of trials preserves conditions
for existing trial ids. Use the same seed/profile when comparing checkpoints or
student and teacher. This reproduces sampled conditions, not necessarily bitwise
identical GPU physics trajectories. `trials.json` saves the entire planned set
of conditions before simulation starts, including metre/radian offsets and seeds.

Reports go to `eval_student/metrics.json` and `eval_teacher/metrics.json`, or
`--output_dir`. They include original command tracking metrics, per-clip
success/progress/failure reasons, missed reference contact bodies per step,
action-change MSE, and synchronized batch inference latency. Camera rendering
and environment stepping are outside the reported policy inference latency.
`num_clips` counts distinct clips and `num_trials` counts trial rows. `clips`
contains one row per trial, including `trial_id`, `trial_seed`, `perturbation` and
`object_pose_offset`. `clip_summary` groups results by clip with successes/trials,
success rate, a 95% Wilson confidence interval, progress and failure counts.
`all_requested_trials_evaluated` is true only when every requested trial finishes;
an interrupted report is partial and must not be treated as the full trial set.

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
