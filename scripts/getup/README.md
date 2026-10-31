# Getup pose calibration

## Checkpoint evaluation and joint diagnostics

From the repository root in the simulation environment:

```bash
python scripts/getup/eval.py \
  --checkpoint outputs/2026-10-31_01-11-28_getup_2xa800_16k/model_2000.pt \
  --num_envs 1 --episodes 1 --seed 42 \
  --output outputs/getup_eval_model2000 --plots
```

The default is headless, deterministic policy inference with randomized training
resets. `--episodes` is the total number across environments, must be at least
`--num_envs`, and is allocated deterministically (e.g. 3 episodes on 2 environments
means quotas 2 and 1). The output directory must be new. Compilation is disabled
for evaluation and optimizer states are not loaded. Training settings are not edited.
Use `env.pose.randomize_joints=False` for the nominal reset.

For video append `--video --rendering_mode performance
env.viewer.resolution='[640,480]'`. Video covers the evaluation loop and is saved
under the output's `videos/`; data-only evaluation avoids rendering. With multiple
environments, the default camera is not a tiled view and completed-quota environments
may continue simulating while other environments finish; their extra episodes are
excluded from the data and metrics. Progress is printed every 50 control steps.

Outputs:

- `metadata.json`, `env.yaml`, `agent.yaml`: joint order, active/mimic mapping,
  checkpoint, seed, sampling rates, tolerances, effective evaluation configuration.
- `rollout_envNNN_episodeNNNN.npz`: post-physics joint position/velocity, final
  position targets (all 53 joints), computed/applied torque estimates, pelvis
  height and episode-relative timestamps at 200 Hz. Runtime hard/soft position,
  velocity and effort limits and initial joint/root states are stored in each file.
  Control timestamps, consecutive success steps and latched success use 50 Hz.
- `joint_summary.csv`: per-episode, per-joint extrema, tracking errors, hard/soft
  position and target-position exceedances, velocity exceedances, sampled duration,
  first occurrence, and estimated torque saturation fraction.
- `episodes.csv`, `metrics.json`: success and final-hold rates, end causes, maximum
  hold streak, final/max pelvis height, completion flags and sample counts.
- `plots/` (with `--plots`): all 53 joints' angle/target/limits and velocity/limits,
  plus pelvis height. Position red lines are hard limits, orange dashed lines soft.

The physics recorder callback runs before Isaac Lab updates ArticulationData caches;
fresh state is read directly from the PhysX view. Control-step results are captured
after reward/success evaluation and before automatic reset, preserving terminal
states. Time zero is stored separately; it is excluded from sampled durations.
Raw excess is always retained. Default reporting tolerances are `1e-4` rad for
position and `1e-3` rad/s for velocity, configurable with `--position_tolerance`
and `--velocity_tolerance`. Durations are sample counts times physics dt; no claim
is made about inter-sample continuous-time extrema. Saturation means estimated
computed torque magnitude reaches the runtime effort limit. Implicit actuator
torques are PD estimates for the just-applied substep, not measured PhysX forces.

Success still requires pelvis strictly above 0.76 m for 50 control steps; success
does not terminate the episode. `time_out` and `success=true` can coexist. Evaluation
does not add action clipping or change reward/termination rules. Partial episodes
on interruption are marked incomplete and excluded from completed success rates.

## Standalone pose inspection

This standalone scene reuses WEAVE's robot, actuators, ground, and mimic mapping.
It does not register a training task or modify training defaults.

Activate the existing simulation environment first (its activation script sets
the Isaac Sim and Isaac Lab paths):

```bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/getup/calibrate.py --pose standing --seconds 30 --output outputs/getup_calibration/standing
python scripts/getup/calibrate.py --pose supine --hold-pose --seconds 30 --output outputs/getup_calibration/supine_geometry
```

Supine uses the approved refined seed from `tasks/getup/pose_cfg.py`: root
height 0.0595360173 m, its normalized wxyz quaternion, and 41 named active
joint angles. Randomization is centered on this seed and mimic states are
derived from their drivers. WEAVE's standing defaults, action offsets and
relative-observation references are preserved. The red sphere is the virtual
head point; head/height reward calibration remains provisional.

`--hold-pose` rewrites the candidate every physics step. Use it only for visual
geometry inspection. Remove it to observe the free base under gravity with
position control holding the candidate joints. This is not a learned policy.

Each run writes `candidate.json` and 50 Hz `measurements.csv`. Outputs contain
pelvis height, head world height, ankle-link heights, head height relative to
the mean ankle-link height, and speeds. Ankle-link origins are not sole contact
points. Body heights alone cannot certify absence of collision penetration.

Edit a copy of `candidate.json` to adjust named joint positions, `root_pos`, or
`root_quat_wxyz`, then load it with `--pose-json PATH`. Quaternion order is wxyz.
The command-line `--root-height` overrides the JSON height. Head offset is set
separately using `--head-offset X Y Z` (torso link frame, meters).

After visually confirming the nominal pose, try:

```bash
python scripts/getup/calibrate.py --pose-json outputs/getup_calibration/supine_geometry/candidate.json --randomize --seed 1 --seconds 10 --output outputs/getup_calibration/random_1
```

Randomization follows HoST: driver positions times U(0.9,1.1), plus
U(-0.1,0.1) rad, clipped to compatible hard limits. Mimic states are derived
from drivers; all initial velocities are zero. Repeat with multiple seeds.

Calibration decisions to record manually after inspection:

- Supine root/joint pose and admissible randomization/contact behavior.
- Head offset and head target relative to the feet.
- Post activation height, separately from the provisional 0.76 m success height.

This script does not select a post gate automatically: static standing alone
cannot establish which gate yields useful getup training. It also does not
validate the 1-second success criterion without a policy holding the robot up.
Use `--headless` for startup/data smoke checks, not visual approval of a pose.

Validation: the earlier 1-second randomized supine run ultimately produced
50 measurement samples, despite an earlier report that it had not completed.
That run used the old 0.30 m airborne candidate, not the current 0.06 m reset.
The subsequent integrated RL smoke test validates reset/history/timeout/PPO,
but is not visual approval of either pose. Root pose now comes from the shared
`tasks/getup/pose_cfg.py`; simulation dt, render interval and GPU patch capacity
match WEAVE. Profiling flags `--profile` and `--no-marker` remain available.

## Actual RL reset audit

`audit_reset.py` calls the training environment's reset, checks state/history
invariants and partial resets, saves still images, and runs four short dynamics
cases. It adds a diagnostic contact sensor but does not edit training settings.

```bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/getup/audit_reset.py --headless --num-envs 16 --samples 1024 --seconds 1 --output outputs/getup_reset_audit
```

Use a separate output directory for subsequent runs. `--no-images` skips still
images. `analyze_reset_audit.py` analyzes source URDF collision geometry;
`check_imported_collisions.py --usd PATH` additionally checks converted USD
collision shapes and requires Isaac Sim's pxr libraries. Both geometry checks
exclude PhysX cooking tolerances and contact/rest offsets; neither certifies
absence of self-collision.

The 2026-10-30 audit found initial ground overlap with the current 0.06 m root
height, despite passing state invariants and short dynamics checks. See
`outputs/getup_reset_audit/REPORT.md` for measured results and scope. No reset
parameters were changed by this audit.

## Per-joint sensitivity at root 0.074 m

`sensitivity_reset.py` changes root height only in its diagnostic environment.
It generates shared samples through the actual RL reset, then runs 41
single-joint random tests, 41 deterministic scans, 21 leg groups, 41 paired
restore-one-joint ablations, and baseline/full controls. Mimic states follow
their drivers. It also runs 16 selected poses for 1 second holding their reset
targets. Geometry uses imported USD colliders, with a numerical regression
against the earlier independent USD audit.

```bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/getup/sensitivity_reset.py --headless --root-height .074 --usd PATH_TO_CONVERTED_ROBOT_USD --output outputs/getup_reset_sensitivity_074_new
python scripts/getup/report_sensitivity.py --output outputs/getup_reset_sensitivity_074_new
```

The output directory must not already exist. The recorded run is under
`outputs/getup_reset_sensitivity_074`; its REPORT.md describes the scope,
scan directions, group comparisons and paired ablation results. This test
does not select new per-joint randomization ranges automatically.

## Supine seed exploration

The recorded `outputs/getup_supine_exploration/REPORT.md` compares unchanged
baselines, a HoST-inspired mapped reference, three local solutions, and a
refined solution using the current 68 imported USD collision shapes.
`explore_supine.py` searches symmetric poses, anchoring height to posterior
pelvis/torso geometry; `refine_supine.py` adds failure cases and range corners,
then evaluates all candidates on a fresh shared 1024-sample table.
`validate_supine_candidate.py` compares the refined pose and its five lowest
random samples with baselines in PhysX. `report_supine.py` exports named full
joint states, comparison plots, mapping information and force statistics.

The refined candidate passed all 1024 fresh static geometry samples, but its
short settling rollout has a larger peak torso contact force than baseline.
It uses a flexed waist and posterior pelvis support. The user subsequently
approved this refined pose as the training seed; the reset and calibration
script now share it through `GetupPoseCfg.seed_joint_pos`. Historical reports
describe the state before that promotion. Sampling success is not a proof over
the continuous randomization domain. Exact candidate states are stored in
`*_fresh.json`, with full joint order and schema in `fresh_results.json`.
