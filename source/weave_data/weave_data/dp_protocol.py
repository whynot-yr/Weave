"""Local HTTP protocol, following Arena's JSON + base64 RGB transport pattern.

No pickle, no simulator or LeRobot imports. Wire actions are unnormalized.
"""

import base64
import json

import numpy as np

from .dp_codec import PROTOCOL, active_joint_indices

WIRE_VERSION = "weave-dp-http-v1"
MAX_BODY_BYTES = 1024 * 1024
OFFSETS = (0, 5, 10, 15, 20)


def json_bytes(payload):
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")


def read_json(raw):
    def invalid(value):
        raise ValueError(f"Invalid JSON constant {value}")

    if len(raw) > MAX_BODY_BYTES:
        raise ValueError("HTTP payload too large")
    value = json.loads(raw, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def array(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")
    return value


def encode_rgb(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3):
        raise ValueError("RGB must be uint8[224,224,3]")
    return {"shape": [224, 224, 3], "dtype": "uint8", "data_b64": base64.b64encode(rgb.tobytes()).decode("ascii")}


def decode_rgb(payload):
    if not isinstance(payload, dict) or payload.get("shape") != [224, 224, 3] or payload.get("dtype") != "uint8":
        raise ValueError("Invalid RGB metadata")
    raw = base64.b64decode(payload["data_b64"], validate=True)
    if len(raw) != 224 * 224 * 3:
        raise ValueError("Invalid RGB byte count")
    return np.frombuffer(raw, dtype=np.uint8).reshape(224, 224, 3).copy()


def validate_identity(payload, inference=False):
    if payload.get("protocol_version") != WIRE_VERSION:
        raise ValueError("HTTP protocol version mismatch")
    session = payload.get("session_id")
    if not isinstance(session, str) or not 1 <= len(session) <= 128:
        raise ValueError("Invalid session_id")
    for key in ("episode_id", "request_id", "control_step") if inference else ("episode_id",):
        if type(payload.get(key)) is not int or payload[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")


def validate_health(health):
    if health.get("protocol_version") != WIRE_VERSION or health.get("status") != "ready":
        raise ValueError("DP service is not ready or protocol mismatch")
    if health.get("state_shape") != [88] or health.get("action_shape") != [40, 125]:
        raise ValueError("DP shape mismatch")
    if health.get("image_shape") != [224, 224, 3] or health.get("n_obs_steps") != 1:
        raise ValueError("Image/history protocol mismatch")
    p = health["data_protocol"]
    if p.get("protocol") != PROTOCOL or not p.get("complete"):
        raise ValueError("Incompatible/incomplete checkpoint data protocol")
    if p.get("position_mode") != "direct" or p.get("quaternion_order") != "wxyz":
        raise ValueError("Pose convention mismatch")
    if p.get("rotation_6d") != "first_two_matrix_columns_row_major":
        raise ValueError("Rotation layout mismatch")
    if p.get("pose_frame") != "current_actual_pelvis_at_window_start":
        raise ValueError("Reference anchor convention mismatch")
    indices = active_joint_indices(p["joint_names"], p["action_names"])
    if indices != p["active_joint_indices"]:
        raise ValueError("Invalid checkpoint active joint mapping")
    if len(p["body_names"]) != 54 or len(set(p["body_names"])) != 54:
        raise ValueError("Invalid body/contact ordering")
    if not np.isfinite(p["fps"]) or p["fps"] <= 0:
        raise ValueError("Invalid reference fps")
    return p
