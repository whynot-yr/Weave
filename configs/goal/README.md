# Goal-conditioned PPO+BC student

This task uses RSL-RL's `OnPolicyRunner` and `RolloutStorage` with the historical
`MuonPPODistill` loss. The depth task remains available independently.

Student observations, in order:

- `goal_robot_pose`: sampled-goal joint positions, anchor position and rot6d.
- `goal_object_pose`: sampled-goal object position and rot6d.
- `object_state`: live object observations from the base HOI task.
- `robot_proprio`: live robot proprioception from the base HOI task.

Goal poses are expressed in the current actual robot anchor frame. The default
`random_future` mode samples a strictly future frame and resamples when reference
time reaches that frame (not when the actual robot reaches the target pose).
The optional `terminal` mode fixes the target at `T = clip_length - 1`, independent
of the RSI starting frame. Clip resampling also updates the target.
Teacher and critic observations still follow the current reference timeline.

The base task's rewards, terminations, actions and simulation settings are
inherited. Actor/critic observation statistics update as in the historical
implementation; the goals are not restricted to identity normalization.

Before training, match the configuration in
`tasks/hoi/agents/rsl_rl_goal_cfg.py` (or YAML overrides) to the teacher's
`params/agent.yaml`: encoder specs, latent size, expansion, actor/teacher/critic
widths and block counts, observation groups, normalization flags and reference
future offsets. The defaults use the historical 512-wide example, not an
automatically inferred teacher architecture.

```bash
/data/g1_hoi_ws/sim51/bin/python scripts/rsl_rl/train.py \
  --task G1-Inspire-HOI-Goal-v0 \
  --config-dir ./configs/goal --config-name train \
  --load_run /absolute/path/to/teacher/run \
  --checkpoint model_XXXX.pt
```

The standard training entry loads the teacher tracker checkpoint and initializes
the student backbone and shared encoders. Goal encoders start fresh. Outputs use
the base task's existing action dimensions, order and control interpretation.
TensorBoard includes the standard environment reward/termination metrics.
Checkpoints use the standard runner format, not the former v5 DAgger schema.

As in the historical code, the BC annealing counter is algorithm-local and is
not automatically restored by the standard runner checkpoint. This change does
not add a custom resume mechanism.

## Evaluate with goal visualization

```bash
/data/g1_hoi_ws/sim51/bin/python scripts/rsl_rl/eval.py \
  --task G1-Inspire-HOI-Goal-v0 \
  --config-dir ./configs/goal --config-name eval \
  --checkpoint /absolute/path/to/student/model_XXXX.pt --real-time
```

The evaluation config inherits the motion files from `train.yaml`; override
`env.commands.motion.motion_files` for a different single-object evaluation set.
All clips are evaluated, while `goal_vis_env_ids: [0]` limits command markers to
environment 0. Use `null` for all environments or another list of environment IDs.
This filter does not change the evaluated clips or the viewer camera.

## Evaluate from random reference states

Use `eval_random` to evaluate each clip from an independently sampled reference
frame instead of frame 0:

```bash
source /data/g1_hoi_ws/sim51/bin/activate
/data/g1_hoi_ws/sim51/bin/python scripts/rsl_rl/eval.py \
  --task G1-Inspire-HOI-Goal-v0 \
  --config-dir ./configs/goal --config-name eval_random \
  --checkpoint /absolute/path/to/student/model_XXXX.pt \
  --seed 42 --output_dir outputs/eval_random_seed42 --real-time
```

For a clip with frames `0, ..., T`, the start `t` is sampled uniformly from
`[0, T-1]`, excluding the terminal frame so a strictly future goal exists.
Robot root and object poses and velocities, and joint positions and velocities,
are restored from reference frame `t` (joint positions retain the existing limit
clamping). The first goal is sampled uniformly from `[t+1, T]`. When reference
time reaches goal frame `m < T`, a new goal is sampled from `[m+1, T]`; the final
goal is held until termination. Switching goals does not reset simulation state
and does not wait for the actual robot to reach the target. A single-frame clip
uses frame 0 for both start and goal.

The existing `eval` config keeps `eval_start_mode: zero`. You can also select
random starts with `env.commands.motion.eval_start_mode=random` on an existing
evaluation command. This setting only affects evaluation; training still uses
`rsi`. `eval_random` explicitly selects `goal_sampling_mode: random_future`.

Each environment scores only its first episode. `metrics.json` records the seed
and start mode, plus each clip's `start_frame`, original `length`, and
`remaining_length`. Progress is measured against the remaining episode, so
running from a random start to the clip end yields 100%. With the current
termination-before-command-update order, the terminal reference frame contributes
one final control step: a clip of length `L` starting at `t` has `L-t` remaining
steps. Error averages cover only the evaluated suffix; aggregate errors retain
the existing step weighting. Success requires reaching the clip end without any
failure condition, including failures coinciding with clip end. Fixed seeds are
reproducible under the same execution conditions; use multiple seeds to compare
performance across starting states. These suffix evaluations measure a different
task from completing entire clips from frame 0.

## Initialize from A and follow goals from B

Use `eval_cross_clip` to restore the robot and object from clip `A` at frame `t`,
then follow the reference and sampled future goals of another clip `B`, starting
at the same frame index `t`:

```bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/rsl_rl/eval.py \
  --task G1-Inspire-HOI-Goal-v0 \
  --config-dir ./configs/goal --config-name eval_cross_clip \
  --checkpoint /absolute/path/to/student/model_XXXX.pt \
  --seed 42 --output_dir outputs/eval_cross_clip_seed42 --headless
```

The environment-to-initialization-clip assignment stays fixed: without
`--num_envs`, each loaded clip serves as `A` once; `--num_envs 1` uses clip 0 as
`A`. Each reset independently samples `B != A` uniformly from clips of the same
object that have a frame after `t`. Different environments may choose the same
`B`; a different variant of the same motion counts as another clip. `B` stays
fixed throughout that episode. Goal frames are sampled from `B[t+1, T_B]`, then
`B[m+1, T_B]` when reference time reaches goal frame `m`. Goal switches do not
reset simulation state or select another clip.

For random initialization, `t` is sampled uniformly from A's usual non-terminal
start range restricted to frames with at least one compatible `B`. Thus if A has
`L_A` frames, the number of possible starts is
`min(max(L_A-1, 1), max_{B != A, same object}(L_B-1))`. Unequal clip lengths can
shorten this range; B's time is never silently clamped or remapped. A single-frame
`A` can initialize at frame 0 if a compatible B has at least two frames. Missing
compatible clips cause an explicit error. To initialize at frame 0, append
`env.commands.motion.eval_start_mode=zero`. The existing `eval` and `eval_random`
configs retain `eval_clip_mode: same_clip`; training is unaffected by this setting.

Reference observations, tracking errors, failure conditions, and clip-end
termination all use B. Reference errors are recorded and checked from the first
step: there is no first-goal grace period. Goals still switch on reference time
in the default `eval_success_mode: reference` mode. Coordinates are not aligned
between A and B; each environment's origin is added to the dataset coordinates.

`metrics.json` records A/B IDs, names, initialization frame, reference length,
first goal, executed steps, and measured steps. Only the first episode per
environment is scored. The number of trials is not the number of unique B clips.

## Object goal success evaluation

Use `--config-name eval_cross_clip_task` for `eval_success_mode: goal_success`.
A/B selection and initialization are unchanged (random by default; append
`env.commands.motion.eval_start_mode=zero` for A[0]). All reference error metrics
are diagnostic and recorded from the first step, but reference-based failures
and clip-end termination are disabled. Training and checkpoint inputs are unchanged.

The current B[m] goal stays fixed until the actual object is within 0.10 m and
15 degrees of its full quaternion orientation for 0.5 consecutive seconds.
Leaving this region resets the hold counter. Then a new goal is drawn uniformly
from [m+1, T_B]. Reaching and holding the final goal succeeds. Reference time
continues independently and clamps at the final frame; it never switches goals
in this mode. The total episode budget is 20 seconds, independent of clip length.

Failure is timeout or an absolute robot fall check: root height below 0.35 m
relative to the environment origin, or root tilt above 75 degrees. These are
configurable initial thresholds; calibrate them for legitimate crouching poses.
A fall takes precedence over simultaneous success. Goal success is object pose
reaching, not proof of release, support, grasp correctness, or low object speed.

Settings live under `env.commands.motion`: `goal_position_threshold`,
`goal_rotation_threshold_deg`, `goal_hold_seconds`, `fallen_height_threshold`,
`fallen_tilt_threshold_deg`. Budget: `env.episode_length_s`.

Task reports contain `success_rate`, `goal_reach_rate`, per-trial elapsed time,
completed goal history (frame and cumulative completion time), current goal,
goals sampled/reached, and final object position/rotation errors. Rotation error
values are radians. Terminal snapshots are captured before automatic reset.
`progress_rate` is omitted because reference-clock progress is not task progress.
Completed intermediate goals contribute only to the auxiliary goal reach rate;
only completing the last B frame counts as whole-episode success.

## Goal markers and recording

Large XYZ frames show the sampled root (pelvis) and object goals. A cyan dot marks
the root goal and a magenta dot marks the object goal; small frames show actual
root/object poses. They read the same world-space targets used by observations,
including updates in `random_future` mode. Set `show_reference_bodies: true` to
also show the current timeline reference and actual tracked bodies.

Set `env.commands.motion.debug_vis=false` for evaluation without markers, and
use `--headless` when no interactive viewer is needed. Evaluation exits once all
clips finish. Environments may reset after finishing; the evaluator scores only
their first episode, while markers continue showing their current goals.

Add `--video` to record the viewer camera, including visible goal markers, walls,
and lighting. Videos are saved to `<output_dir>/videos/eval.mp4`, alongside
`metrics.json`; without `--output_dir`, the directory is `<checkpoint_dir>/eval`.
An existing `eval.mp4` is overwritten. Recording starts after the evaluation
reset and captures one frame per control step (50 FPS with the default settings).
`--video_length 0` (default) records until evaluation ends; a positive value caps
the number of recorded steps without shortening evaluation. Frames already
captured are also saved on interruption. `--headless --video` enables camera
rendering automatically; `--real-time` is optional and does not change video FPS.
With `--num_envs 1`, only clip 0 is evaluated and recorded. Camera placement and
resolution follow `env.viewer`.
