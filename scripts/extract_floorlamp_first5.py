import os
import numpy as np


# ============================================================
# 1. 路径
# ============================================================

SRC = "/home/whynot/Downloads/floorlamp_33clip_var5.npz"

DST = "/data/g1_hoi_ws/Weave/data/train/floorlamp_033_var5.npz"


# ============================================================
# 2. 读取原始 NPZ
# ============================================================

data = np.load(SRC, allow_pickle=True)

print("Original keys:")
for key in data.files:
    print(" ", key)

print()


# ============================================================
# 3. 读取 motion metadata
# ============================================================

motion_names = data["motion_names"]
motion_lengths = data["motion_lengths"]

num_motions = len(motion_names)
total_frames = int(np.sum(motion_lengths))

print("Original number of motions:", num_motions)
print("Original total frames:", total_frames)
print()


# ============================================================
# 4. 打印前 10 条 motion，先确认顺序
# ============================================================

print("First 10 motions:")

for i in range(min(10, num_motions)):
    name = motion_names[i]

    if isinstance(name, bytes):
        name = name.decode()

    print(
        f"{i:3d}: "
        f"{name:35s} "
        f"length={int(motion_lengths[i])}"
    )

print()


# ============================================================
# 5. 选择最前面的 5 条 motion
#
# index 0~4:
#
# sub17_floorlamp_033_v00
# sub17_floorlamp_033_v01
# sub17_floorlamp_033_v02
# sub17_floorlamp_033_v03
# sub17_floorlamp_033_v04
# ============================================================

selected_motion_ids = np.array(
    [0, 1, 2, 3, 4],
    dtype=np.int64,
)


# ============================================================
# 6. 计算每条 motion 在大 frame buffer 中的起始位置
#
# 例如：
#
# motion_lengths:
# [330, 330, 330, ...]
#
# starts:
# [0, 330, 660, ...]
# ============================================================

motion_starts = np.concatenate(
    [
        np.array([0], dtype=np.int64),
        np.cumsum(motion_lengths[:-1]),
    ]
)


# ============================================================
# 7. 找出选中的 5 条 motion 对应哪些 frame
# ============================================================

selected_frame_indices = []

print("Selected motions:")

for motion_id in selected_motion_ids:

    start = int(motion_starts[motion_id])

    length = int(motion_lengths[motion_id])

    end = start + length

    name = motion_names[motion_id]

    if isinstance(name, bytes):
        name = name.decode()

    print(
        f"motion_id={motion_id:3d}  "
        f"{name:35s}  "
        f"frames=[{start}:{end})  "
        f"length={length}"
    )

    selected_frame_indices.extend(
        range(start, end)
    )


selected_frame_indices = np.asarray(
    selected_frame_indices,
    dtype=np.int64,
)

print()

print(
    "Selected total frames:",
    len(selected_frame_indices),
)

print()


# ============================================================
# 8. 数据字段分类
#
# clip-level:
#     每条 motion 一个值
#
# frame-level:
#     每一 frame 一个值
#
# global:
#     整个文件共享，例如 fps
# ============================================================

clip_level_keys = [
    "motion_lengths",
    "motion_names",
    "object_names",
]

frame_level_keys = [
    "joint_pos",
    "joint_vel",

    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",

    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",

    "contact_label",
]


# ============================================================
# 9. 开始裁剪
# ============================================================

output = {}


# -------------------------
# clip-level 数据
# -------------------------

for key in clip_level_keys:

    if key not in data.files:
        raise KeyError(
            f"Missing clip-level key: {key}"
        )

    output[key] = data[key][selected_motion_ids]


# -------------------------
# frame-level 数据
# -------------------------

for key in frame_level_keys:

    if key not in data.files:
        raise KeyError(
            f"Missing frame-level key: {key}"
        )

    arr = data[key]

    if arr.shape[0] != total_frames:
        raise ValueError(
            f"{key}: first dimension should be "
            f"{total_frames}, but got {arr.shape[0]}"
        )

    output[key] = arr[selected_frame_indices]


# -------------------------
# global metadata
# -------------------------

if "fps" in data.files:
    output["fps"] = data["fps"]


# ============================================================
# 10. 保存新的 NPZ
# ============================================================

os.makedirs(
    os.path.dirname(DST),
    exist_ok=True,
)

np.savez_compressed(
    DST,
    **output,
)


# ============================================================
# 11. 重新打开，进行完整一致性检查
# ============================================================

check = np.load(
    DST,
    allow_pickle=True,
)

print("=" * 70)
print("Saved file:")
print(DST)
print("=" * 70)

print()

print("motion_names:")

for i, name in enumerate(check["motion_names"]):

    if isinstance(name, bytes):
        name = name.decode()

    print(
        f"{i}: {name}"
    )

print()

print(
    "motion_lengths:",
    check["motion_lengths"],
)

print(
    "number of motions:",
    len(check["motion_names"]),
)

print(
    "sum(motion_lengths):",
    int(np.sum(check["motion_lengths"])),
)

print()


# ============================================================
# 12. 检查所有 frame-level array 是否长度一致
# ============================================================

expected_frames = int(
    np.sum(check["motion_lengths"])
)

print(
    "Expected frame count:",
    expected_frames,
)

print()

for key in frame_level_keys:

    arr = check[key]

    print(
        f"{key:22s} "
        f"shape={arr.shape}"
    )

    assert arr.shape[0] == expected_frames, (
        f"{key} has wrong frame count"
    )


# ============================================================
# 13. 最终 sanity check
# ============================================================

assert len(check["motion_names"]) == 5
assert len(check["motion_lengths"]) == 5
assert len(check["object_names"]) == 5

assert expected_frames == check["joint_pos"].shape[0]
assert expected_frames == check["body_pos_w"].shape[0]
assert expected_frames == check["object_pos_w"].shape[0]
assert expected_frames == check["contact_label"].shape[0]

print()
print("All checks passed.")
print("New NPZ is ready for training.")