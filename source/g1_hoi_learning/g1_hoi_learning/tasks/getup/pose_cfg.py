"""Shared calibration/reset definitions. Pending values are explicitly provisional."""
from isaaclab.utils import configclass


@configclass
class GetupPoseCfg:
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
