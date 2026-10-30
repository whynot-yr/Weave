# Getup pose calibration

This standalone scene reuses WEAVE's robot, actuators, ground, and mimic mapping.
It does not register a training task or modify training defaults.

Activate the existing simulation environment first (its activation script sets
the Isaac Sim and Isaac Lab paths):

```bash
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/getup/calibrate.py --pose standing --seconds 30 --output outputs/getup_calibration/standing
python scripts/getup/calibrate.py --pose supine --hold-pose --root-height 0.06 --seconds 30 --output outputs/getup_calibration/supine_geometry
```

The supine candidate uses WEAVE's default joint pose rotated -90 degrees about
world Y. Its root height of 0.06 m is a starting point for inspection, not an
approved reset pose. The red sphere is the virtual head point; its default
offset is derived from the URDF head COM, not an approved reward target.

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
