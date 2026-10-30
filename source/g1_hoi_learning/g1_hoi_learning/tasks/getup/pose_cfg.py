"""Shared calibration/reset definitions. Pending values are explicitly provisional."""
from isaaclab.utils import configclass


@configclass
class GetupPoseCfg:
    root_pos: tuple = (0.0, 0.0, 0.06)
    root_quat: tuple = (0.70710678, 0.0, -0.70710678, 0.0)  # wxyz
    # Nominal joint angles are WEAVE's default_joint_pos, never a new action offset.
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
