"""Named getup seeds; head/height reward calibration remains provisional."""
from isaaclab.utils import configclass


@configclass
class SupineSeedCfg:
    # Approved refined supine seed; independent of standing/action offsets.
    root_pos: tuple = (0.0, 0.0, 0.05953601732234268)
    root_quat: tuple = (0.7066905208230118, 0.0, -0.7075227966496206, 0.0)  # wxyz
    seed_joint_pos: dict[str, float] = {
        'left_hip_pitch_joint': -0.45000000000000007,
        'right_hip_pitch_joint': -0.45000000000000007,
        'waist_yaw_joint': 0.0,
        'left_hip_roll_joint': 0.0,
        'right_hip_roll_joint': 0.0,
        'waist_roll_joint': 0.0,
        'left_hip_yaw_joint': 0.0,
        'right_hip_yaw_joint': 0.0,
        'waist_pitch_joint': 0.39481580844104863,
        'left_knee_joint': 0.45,
        'right_knee_joint': 0.45,
        'left_shoulder_pitch_joint': -0.1,
        'right_shoulder_pitch_joint': -0.1,
        'left_ankle_pitch_joint': -0.2,
        'right_ankle_pitch_joint': -0.2,
        'left_shoulder_roll_joint': 0.25,
        'right_shoulder_roll_joint': -0.25,
        'left_ankle_roll_joint': 0.0,
        'right_ankle_roll_joint': 0.0,
        'left_shoulder_yaw_joint': 0.0,
        'right_shoulder_yaw_joint': 0.0,
        'left_elbow_joint': 0.7999999999999997,
        'right_elbow_joint': 0.7999999999999997,
        'left_wrist_roll_joint': 0.0,
        'right_wrist_roll_joint': 0.0,
        'left_wrist_pitch_joint': 0.0,
        'right_wrist_pitch_joint': 0.0,
        'left_wrist_yaw_joint': 0.0,
        'right_wrist_yaw_joint': 0.0,
        'L_index_proximal_joint': 0.0,
        'L_middle_proximal_joint': 0.0,
        'L_pinky_proximal_joint': 0.0,
        'L_ring_proximal_joint': 0.0,
        'L_thumb_proximal_yaw_joint': 0.0,
        'R_index_proximal_joint': 0.0,
        'R_middle_proximal_joint': 0.0,
        'R_pinky_proximal_joint': 0.0,
        'R_ring_proximal_joint': 0.0,
        'R_thumb_proximal_yaw_joint': 0.0,
        'L_thumb_proximal_pitch_joint': 0.0,
        'R_thumb_proximal_pitch_joint': 0.0,
    }


@configclass
class ProneSeedCfg:
    # Approved refined_arms candidate; keep WEAVE action/standing offsets separate.
    root_pos: tuple = (0.0, 0.0, 0.060718337089254)
    root_quat: tuple = (0.6968723858985278, 0.0, 0.7171951462273662, 0.0)  # wxyz
    seed_joint_pos: dict[str, float] = {
        'left_hip_pitch_joint': 0.4415855047906164,
        'right_hip_pitch_joint': 0.4415855047906164,
        'waist_yaw_joint': 0.0,
        'left_hip_roll_joint': 0.0,
        'right_hip_roll_joint': 0.0,
        'waist_roll_joint': 0.0,
        'left_hip_yaw_joint': 0.0,
        'right_hip_yaw_joint': 0.0,
        'waist_pitch_joint': -0.3441417467181422,
        'left_knee_joint': 0.2500303156992445,
        'right_knee_joint': 0.2500303156992445,
        'left_shoulder_pitch_joint': 0.2998624533025993,
        'right_shoulder_pitch_joint': 0.2998624533025993,
        'left_ankle_pitch_joint': 2.0210466164886303e-05,
        'right_ankle_pitch_joint': 2.0210466164886303e-05,
        'left_shoulder_roll_joint': 0.9804392427619013,
        'right_shoulder_roll_joint': -0.9804392427619013,
        'left_ankle_roll_joint': 0.0,
        'right_ankle_roll_joint': 0.0,
        'left_shoulder_yaw_joint': -1.733184017299804,
        'right_shoulder_yaw_joint': 1.733184017299804,
        'left_elbow_joint': 2.0763183383180706,
        'right_elbow_joint': 2.0763183383180706,
        'left_wrist_roll_joint': 0.7579787270424456,
        'right_wrist_roll_joint': -0.7579787270424456,
        'left_wrist_pitch_joint': -0.6065477948911066,
        'right_wrist_pitch_joint': -0.6065477948911066,
        'left_wrist_yaw_joint': 0.0,
        'right_wrist_yaw_joint': 0.0,
        'L_index_proximal_joint': 0.0,
        'L_middle_proximal_joint': 0.0,
        'L_pinky_proximal_joint': 0.0,
        'L_ring_proximal_joint': 0.0,
        'L_thumb_proximal_yaw_joint': 0.0,
        'R_index_proximal_joint': 0.0,
        'R_middle_proximal_joint': 0.0,
        'R_pinky_proximal_joint': 0.0,
        'R_ring_proximal_joint': 0.0,
        'R_thumb_proximal_yaw_joint': 0.0,
        'L_thumb_proximal_pitch_joint': 0.0,
        'R_thumb_proximal_pitch_joint': 0.0,
    }


@configclass
class GetupPoseCfg:
    reset_mode: str = "mixed"  # supine, prone, mixed
    prone_probability: float = 0.5
    supine: SupineSeedCfg = SupineSeedCfg()
    prone: ProneSeedCfg = ProneSeedCfg()
    randomize_joints: bool = True
    joint_scale_range: tuple = (0.9, 1.1)
    joint_offset_range: tuple = (-0.1, 0.1)
    head_offset: tuple = (0.0092305, 0.000299, 0.405869)
    head_target: float = 1.1716  # candidate from default-standing geometry; NOT calibrated
    head_margin: float = 1.0
    style_height_gate: float = 0.45  # HoST ground candidate
    post_height_gate: float = 0.65  # HoST ground candidate; independent of success
    standing_height_target: float = 0.75  # HoST ground candidate
    success_height: float = 0.76
    success_hold_s: float = 1.0
    calibration_verified: bool = False
