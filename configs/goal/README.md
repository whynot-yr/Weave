# Terminal-goal PPO+BC student

This task uses RSL-RL's `OnPolicyRunner` and `RolloutStorage` with the historical
`MuonPPODistill` loss. The depth task remains available independently.

Student observations, in order:

- `goal_robot_pose`: final-frame joint positions, anchor position and rot6d.
- `goal_object_pose`: final-frame object position and rot6d.
- `object_state`: live object observations from the base HOI task.
- `robot_proprio`: live robot proprioception from the base HOI task.

Goal poses are expressed in the current actual robot anchor frame. The world
target stays fixed within a selected clip; its relative representation changes
as the robot moves. `T = clip_length - 1` is independent of the RSI starting
frame. Clip resampling updates the target, including on training rollover.
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
