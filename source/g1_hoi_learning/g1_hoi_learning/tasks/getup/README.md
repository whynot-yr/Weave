# WEAVE Getup V1

Task ID: `G1-Inspire-Getup-v0`. Uses WEAVE's G1 Inspire URDF, implicit actuators,
software mimic actions, SimBaV2 components, Muon/AdamW, and the installed RSL-RL
`OnPolicyRunner`. The HOI task is unchanged. No motion files, objects, teacher,
phase state machine, external pushes, or extra dynamics randomization are used.

## Configuration

- Observation: `[base_ang_vel_b(3), projected_gravity_b(3), joint_pos_rel(53),
  joint_vel_rel(53), pelvis_height(1), previous_action(41)]`.
- Three frames including the current frame, oldest first: **462D**. Reset repeats
  the first frame. Actor and critic use the same noiseless observation group.
- History is explicitly owned by `GetupState`, rather than ObservationManager's
  history. This allows terminal snapshots before auto-reset without advancing
  history twice during logging or observation queries.
- Actor and four independent critics: MLP `462 -> 256 -> 256` (SiLU), WEAVE SimBa
  width 2048, two blocks, expansion 1. Critic outputs concatenate to `[N,4]`.
- Action: 41D diagonal Gaussian, initial std 0.8. WEAVE's default offset and
  scale 1.0 are retained. Reset does not change the default joint offsets.
- Reset: root `(0,0,0.06)` plus environment origin, quaternion wxyz
  `(sqrt(.5),0,-sqrt(.5),0)`, WEAVE default joint pose, zero velocities.
  Driver positions use HoST `q_seed * U(.9,1.1) + U(-.1,.1)`, clipped to hard
  limits including compatible mimic limits. Mimic positions derive from drivers.
- 200 Hz physics, 50 Hz control, 10 s timeout. Joint velocity termination checks
  all 53 joints against 300 rad/s. Base speed termination uses 3D L2 norm >20 m/s.
- Success: pelvis >0.76 m continuously for 50 control steps; interrupted streaks
  reset. Episode success latches until reset; no success termination.
- Upper-body posture: 17 joints (waist + shoulders/elbows/wrists), no fingers.

## Rewards and PPO

Group order is always **task/style/regu/post**. Task is the HoST head/upright
product; style retains original height gates, knee=-0.25, stumble=0, and excludes
ankle/ground-parallel. Regularization uses the nine Ground terms over 41 drivers.
Post uses a separate pelvis-height gate. Head height is relative to the average
of the two ankle-roll link origins, not world height or sole height.

Task is per-step, without dt; all other groups multiply control dt **once**.
Each critic learns its unweighted group return. GAE is computed per group,
standardized over the rollout per group, then actor advantage is weighted by
`[2.5,1,0.1,1]`. No second advantage normalization. Value losses average over
batch and four critics; optimizer, clipping, KL adaptation follow WEAVE.
RND, symmetry augmentation and recurrent policies are intentionally unsupported
in this V1 multi-critic class.

The scalar reward is the weighted sum for runner logging. PPO consumes
`extras['reward_groups']`. Timeout bootstrap uses the captured **terminal**
462D observation, not reset observations; real failures suppress bootstrapping,
even if timeout happens in the same step.

Logs include individual terms (Isaac Lab episode normalization), raw group
returns (`Getup_Return/*`), episode/final-hold success, and pre-update value MSE
and explained variance per critic. Term time scales are already applied in the
custom manager; never apply dt a second time outside it.

## Pending calibration

`pose_cfg.py` is the single location for reset root pose and calibration values.
`calibration_verified=False` is deliberate. Current candidates are:

| Parameter | Candidate |
|---|---:|
| root height | 0.06 m (approved V1 value; collision check still pending) |
| torso-local head offset | (0.0092305, 0.000299, 0.405869) m |
| relative head target | 1.1716 m (initial geometric measurement) |
| style height gate | 0.45 m (HoST Ground) |
| post height gate | 0.65 m (HoST Ground) |
| post standing height target | 0.75 m (HoST Ground) |
| success height | 0.76 m (provisional) |

Passing a numerical smoke test does not approve collision geometry or prove
that these heights yield successful learning. Validate nominal and randomized
poses visually; confirm the head point in standing and supine views. The assets
retain WEAVE's disabled self-collision setting. Implicit-actuator torque rewards
use Isaac Lab's `applied_torque` estimate.

## Run

From repository root:

```bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/getup/smoke.py --headless
python -m pytest source/g1_hoi_learning/tests/test_getup.py -q
```

The integration smoke test uses two environments and smaller test networks,
checks terminal/reset histories, performs MuonPPO updates, and checks the native
RSL-RL runner plus checkpoint restore. Training architecture stays 2048-wide.

Short standard-entry validation (no motion arguments required):

```bash
python scripts/rsl_rl/train.py --task G1-Inspire-Getup-v0 \
  --num_envs 16 --max_iterations 1 agent.policy.compile=False
```

After calibration, choose a parallel environment count that fits the GPU and
increase iterations. Initial episode-length randomization is disabled for this
task only. Play uses the existing `scripts/rsl_rl/play.py` with this task ID and
the getup checkpoint. Exported actor input is 462D; the caller must maintain the
same history order and default-offset action mapping.

## Verified implementation checks

- Six CPU regression tests pass: group GAE/weights, independent critic gradients,
  timeout terminal-value bootstrap, history/continuous success, mixed reward dt
  scaling, and normalizer checkpoint reload after inference-mode collection.
- Two-environment Isaac Sim smoke passes, including native OnPolicyRunner learn,
  save/load and identical inference output after reload.
- Standard `scripts/rsl_rl/train.py` completed one update with the **2048-wide**
  architecture (2 environments, 2 rollout steps, 1 epoch/minibatch, compilation
  disabled), saved `model_0.pt`, and exited normally. This is integration evidence,
  not a convergence or performance benchmark. Default 32/5/4 training settings
  and compiled execution have not been benchmarked here.
- Standalone calibration cleanup now clears SimulationContext before closing
  Kit, avoiding the stop callback's render loop. This explains the previous
  non-exiting process; it does not explain all measured physics-step overhead.
