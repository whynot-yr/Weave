"""WEAVE rollout -> LeRobot v3, using Arena's create/add/save/finalize flow.

No Arena import is required. Raw reference poses remain in env-local world axes;
the sample-time pelvis frame can only be chosen when a training window is built.
"""

import json
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from .dp_codec import IMAGE_KEY, PROTOCOL, REFERENCE_DIMS, active_joint_indices, encode_state


def convert_hdf5(source, output, repo_id, episodes=None, success_only=False, use_videos=True):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with h5py.File(source, "r") as f:
        expected = {
            "schema_version": "weave-rollout-v1",
            "quaternion_order": "wxyz",
            "coordinate_frame": "env_local_world_axes",
        }
        for key, value in expected.items():
            if f.attrs.get(key) != value:
                raise ValueError(f"Unsupported {key}: {f.attrs.get(key)!r}")
        metadata = json.loads(f.attrs["metadata_json"])
        indices = active_joint_indices(metadata["joint_names"], metadata["action_names"])
        if len(metadata["body_names"]) != 54:
            raise ValueError("Expected 54 body/contact names")
        pelvis = metadata["body_names"].index("pelvis")
        fps = float(metadata["fps"])
        if not fps.is_integer() or fps <= 0:
            raise ValueError("LeRobot requires a positive integer fps")
        lengths = f["episodes/length"][:]
        selected = list(range(len(lengths))) if episodes is None else list(episodes)
        if not selected or len(set(selected)) != len(selected) or any(e < 0 or e >= len(lengths) for e in selected):
            raise ValueError("Episode selection must contain unique existing episode indices")
        selected = [e for e in selected if lengths[e] > 0 and (not success_only or f["episodes/success"][e])]
        if not selected:
            raise ValueError("No nonempty episodes selected")
        image_shape = tuple(f["camera/head/rgb"].shape[2:])
        if image_shape != (224, 224, 3):
            raise ValueError(f"Expected 224x224 RGB; got {image_shape}")
        features = {
            IMAGE_KEY: {
                "dtype": "video" if use_videos else "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.state": {"dtype": "float32", "shape": (88,), "names": None},
        }
        for key, dim in {
            **{f"reference.{k}": v for k, v in REFERENCE_DIMS.items()},
            "aux.pelvis_pos": 3,
            "aux.pelvis_quat": 4,
        }.items():
            features[key] = {"dtype": "float32", "shape": (dim,), "names": None}
        dataset = LeRobotDataset.create(
            repo_id,
            fps=int(fps),
            root=output,
            features=features,
            robot_type="weave_g1_inspire",
            use_videos=use_videos,
            image_writer_threads=2,
            video_backend="pyav",
        )
        protocol = {
            "protocol": PROTOCOL,
            "dataset_id": str(uuid.uuid4()),
            "complete": False,
            "source": str(Path(source).resolve()),
            "fps": fps,
            "state_dim": 88,
            "action_dim": 125,
            "pose_frame": "current_actual_pelvis_at_window_start",
            "quaternion_order": "wxyz",
            "rotation_6d": "first_two_matrix_columns_row_major",
            "position_mode": "direct",
            "contact_labels": {"-1": "no_contact", "0": "unconstrained", "1": "contact"},
            "joint_names": metadata["joint_names"],
            "action_names": metadata["action_names"],
            "body_names": metadata["body_names"],
            "active_joint_indices": indices,
            "episodes": [],
        }
        protocol_path = output / "meta/weave_protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2))
        try:
            for e in selected:
                length = int(lengths[e])
                if not f["frame/valid"][e, :length].all():
                    raise ValueError(f"Episode {e}: holes in valid frames")
                times = f["frame/sim_time"][e, :length]
                if length > 1 and not np.allclose(np.diff(times), 1 / fps, atol=1e-5):
                    raise ValueError(f"Episode {e}: nonuniform timestamps")

                def tensor(key):
                    value = torch.from_numpy(f[key][e, :length]).float()
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Episode {e}: nonfinite {key}")
                    return value

                pos = tensor("robot/body_pos")[:, pelvis]
                quat = tensor("robot/body_quat")[:, pelvis]
                state = encode_state(quat, tensor("robot/joint_pos"), tensor("robot/joint_vel"), quat[0], indices)
                reference = {
                    "joint_pos": tensor("reference/joint_pos"),
                    "pelvis_pos": tensor("reference/body_pos")[:, pelvis],
                    "pelvis_quat": tensor("reference/body_quat")[:, pelvis],
                    "object_pos": tensor("reference/object_pos"),
                    "object_quat": tensor("reference/object_quat"),
                    "contact": tensor("reference/contact_label"),
                }
                for q in (quat, reference["pelvis_quat"], reference["object_quat"]):
                    if not torch.allclose(q.norm(dim=-1), torch.ones(length), atol=1e-3):
                        raise ValueError(f"Episode {e}: nonunit quaternion")
                if not torch.isin(reference["contact"], torch.tensor([-1.0, 0.0, 1.0])).all():
                    raise ValueError(f"Episode {e}: expected ternary contact labels -1/0/1")
                object_name = f["episodes/object_name"].asstr()[e]
                for t in range(length):
                    frame = {
                        IMAGE_KEY: f["camera/head/rgb"][e, t],
                        "observation.state": state[t].numpy(),
                        "aux.pelvis_pos": pos[t].numpy(),
                        "aux.pelvis_quat": quat[t].numpy(),
                        "task": f"Manipulate {object_name}",
                    }
                    frame.update({f"reference.{k}": v[t].numpy() for k, v in reference.items()})
                    dataset.add_frame(frame)
                dataset.save_episode()
                protocol["episodes"].append(
                    {
                        "episode_index": len(protocol["episodes"]),
                        "source_episode": e,
                        "clip_name": f["episodes/clip_name"].asstr()[e],
                        "length": length,
                        "success": bool(f["episodes/success"][e]),
                    }
                )
                print(f"Converted source episode {e}: {length} frames", flush=True)
        finally:
            dataset.finalize()
        protocol["complete"] = True
        protocol_path.write_text(json.dumps(protocol, indent=2))
    return protocol
