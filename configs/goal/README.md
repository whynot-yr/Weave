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
