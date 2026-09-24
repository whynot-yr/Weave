# WEAVE 高层 Diffusion Policy

DP→PPO 双进程 HTTP 推理、独立仿真入口与运行命令见 [部署说明](DEPLOYMENT.md)。

本目录只新增离线训练/推理工具，**不修改原有 WEAVE PPO、环境、观测或安装脚本**。
LeRobot 作为依赖使用，不修改其源码；HumanoidArena 的写入流程迁移为
`LeRobotDataset.create → add_frame → save_episode → finalize`，无需运行时导入 Arena。

## 文件职责

- `source/weave_data/weave_data/reference_window.py`：从 WEAVE 原有窗口算法独立出来的纯 Torch 实现。
  取样对应 `MotionCommand._refresh_caches`，坐标转换对应 `observations.py` 的四个 future pose 函数。
  原 PPO 仍使用原函数；测试验证二者等价。不是在 LeRobot 内包装/修改窗口转换。
- `dp_codec.py`：88/125 维协议、关节名字映射、参考重新定锚，以及 PPO 的 310/315 维参考组适配。
- `hdf5_converter.py` / `convert_hdf5.py`：HDF5 写入 LeRobot；保存世界轴下原始参考、实际 pelvis 辅助数据及 RGB。
- `dp_dataset.py`：同一 episode 内取 40 帧；整个窗口统一转到观测时刻的实际 pelvis 坐标系。
- `dp_stats.py` / `compute_stats.py`：只在选定训练 episode 上统计**完成转换的窗口**，排除尾部 padding，不解码 RGB。
- `train.py`：将上述 batch 接入原版 LeRobot `DiffusionPolicy`、processor、优化器/调度器预设及 checkpoint API。
  单设备训练入口，不替换 LeRobot 模型或使用全局 monkeypatch；目前不提供 DDP/EMA。
- `dp_policy.py`：读取 checkpoint/归一化器，输出未归一化参考；不自动向 PPO 注入参考。

## 协议

输入是 `observation.images.head` RGB 与 `observation.state` 88 维：
初始 episode heading 校正后的实际 pelvis rot6d 6 + 41 个主动关节 q + 41 个主动关节 dq。
q/dq 使用直接值，按记录的 `action_names` 从完整 53 关节中映射；不能截取前 41 维。
不输入物体真值、上一帧 PPO action 或未来参考。初始 heading 在 episode 内固定。

输出每帧 125 维：

| 切片 | 数据 |
| --- | --- |
| `0:53` | 完整参考关节角 |
| `53:56` | pelvis 参考位置 |
| `56:62` | pelvis 参考 rot6d |
| `62:65` | 物体参考位置 |
| `65:71` | 物体参考 rot6d |
| `71:125` | 54 个参考接触标签 |

位置是相对于**当前实际 pelvis**的直接坐标，不是逐帧位移，不做积分。
四元数 wxyz；rot6d 是旋转矩阵前两列按行展开，不能用两列直接拼接的另一种约定。
接触标签保留 WEAVE 的三态：`-1` 要求不接触、`0` 不约束、`1` 要求接触。
旋转及接触归一化范围固定 `[-1,1]`；其余采用训练数据 min/max，常量维度做最小范围保护。

磁盘中不直接存 `action`，而是 `reference.*` 与 `aux.pelvis_*`：每帧实际 pelvis 不同，
无法通过拼接逐帧局部标签形成正确窗口。`aux` 只参与转换，Dataset 返回模型的键只有
RGB、state、action、action_is_pad。直接使用原版 `lerobot-train` 加载此原始数据不能替代本入口。

默认 50 Hz 数据、单帧观测、40 帧预测；`n_action_steps=40` 保留整个预测供 PPO 查询。
计划执行 20 帧后重规划：第 j 步应查询 chunk 的 `j+[0,5,10,15,20]`，j=0..19。
先使用 `reanchor_reference` 把预测时刻局部参考变换到当前实际 pelvis，再用
`ppo_reference_groups` 生成按特征展平的两个参考组。模型预测仍须另行做闭环与安全验证。

## 安装与运行

当前本地 LeRobot 0.6.2 需要 Python 3.12，与 Sim 的 Python 3.11 分开。
新增独立环境位于 WEAVE 内，不向现有 `sim51` 安装/升级 LeRobot。

### 推荐：保存并复用数据划分

转换完成后，按原始 motion 家族划分，而不是随机拆帧或窗口：

```bash
.venv-dp/bin/python scripts/dp/split_dataset.py \
  datasets/floorlamp_dp outputs/floorlamp_split.json --val-ratio 0.2 --seed 42

.venv-dp/bin/python scripts/dp/compute_stats.py \
  datasets/floorlamp_dp outputs/floorlamp_stats.json --split outputs/floorlamp_split.json

.venv-dp/bin/python scripts/dp/train.py \
  --dataset datasets/floorlamp_dp --stats outputs/floorlamp_stats.json \
  --split outputs/floorlamp_split.json --output outputs/floorlamp_train \
  --device cuda --steps 100000 --batch-size 64
```

工具仅读取 `meta/weave_protocol.json`，不解码视频、不复制或移动数据。
`clip_name` 去掉末尾 `_vNN` 后作为家族，同一家族的所有变体和重复 episode 进入同一集合。
验证家族数量取 `ceil(家族数 × val_ratio)`，并保证两组非空；只有一个家族时拒绝划分。
33 个家族默认分成 26/7。输出记录数据集 UUID、episode 和家族列表、两边的帧数汇总；
加载时检查数据集匹配、完整覆盖、无重复及家族隔离。已有输出文件不会覆盖。

多物体数据可添加 `--by-object` 按物体分别划分，每个物体至少需要两个家族。
新转换器会写入 `object_name`；旧数据缺少该字段时不能使用此选项，但仍可按全局家族划分。
自定义命名不符合 `_vNN` 规则时，先确认家族命名，不能把未知变体误当独立运动。

`compute_stats.py --split` 只读取训练 episode；`train.py --split` 同时读取训练和验证 episode。
统计仍需单独生成，不会由训练自动计算。原有显式 `--episodes` / `--train-episodes` 用法保留，
但不能与 `--split` 混用。训练仍按现有逻辑定期验证和保存 checkpoint，不自动挑选最佳模型。

```bash
cd /data/g1_hoi_ws/Weave
bash scripts/dp/install.sh /data/g1_hoi_ws/lerobot

.venv-dp/bin/python scripts/dp/convert_hdf5.py \
  datasets/floorlamp_rollout_100_seed42/rollouts.h5 \
  datasets/floorlamp_dp --success-only

# 下例只使用转换后 episode 0 走通流程；正式训练应明确选择整个训练集。
.venv-dp/bin/python scripts/dp/compute_stats.py \
  datasets/floorlamp_dp outputs/dp_stats.json --episodes 0

.venv-dp/bin/python scripts/dp/train.py \
  --dataset datasets/floorlamp_dp --stats outputs/dp_stats.json \
  --train-episodes 0 --output outputs/dp_train \
  --device cuda --steps 100000 --batch-size 64

.venv-dp/bin/python -m pytest source/weave_data/tests -q
```

转换默认存视频，读取显式使用 pyav；`--images` 可存图像（更占空间）。
`--episodes` 对转换器是原始 HDF5 索引，对统计/训练是**转换后的 LeRobot episode 索引**，
映射与 clip 名称记录在 `meta/weave_protocol.json`。输出已存在时拒绝覆盖。
转换中断的数据集标记为未完成，不能误用来训练。

先按原始运动划分训练/验证，再计算训练统计。训练入口可指定 `--eval-episodes`；
会拒绝 train/eval episode 重合，也拒绝同一 `clip_name` 去除 `_vNN` 后的运动家族重合。
不能随机按窗口划分，否则相邻帧及同源变体会泄漏。
统计文件绑定 dataset UUID、训练 episode 列表、horizon，参数不匹配会报错。

`--smoke --device cpu --workers 0 --batch-size 2 --steps 2 --save-every 1`
使用小 U-Net、随机初始化 ResNet 和两步推理采样，仅用于测试链路，不代表可用策略。
安装脚本的 CUDA 可用性输出应为 True 才能使用 CUDA；测试环境可能只有 CPU torch。
ImageNet 权重只在正式新训练时下载。

Checkpoint 包含 LeRobot 配置、模型、pre/post processors，以及 optimizer/scheduler/RNG 状态、
WEAVE 协议和训练签名。`--resume outputs/dp_train/step_00010000` 恢复；保持原来的
`--steps` 总预算、batch size、seed、数据划分等参数，可指定新输出目录。
恢复使用 checkpoint 中保存的归一化器，不用新统计覆盖它。

边界：这里只接通离线训练、checkpoint 推理和参考编码适配；没有启动长训练，
也没有改动/接管原 PPO 推理循环、物体观测或机器人执行接口。

## 本地验证

已验证 9 项自动测试：原 WEAVE 位姿函数等价性、窗口截断与 padding、主动关节映射、
旋转投影与重新定锚、图像/视频两种数据格式、训练集统计约束，以及 LeRobot 的
loss/backward、processor 保存加载和 40×125 参考推理。

另用现有 floorlamp HDF5 的 episode 0 完成 360 帧转换及两步 CPU 冒烟训练。
抽查 5 个真实时刻，生成的 PPO 参考与 HDF5 中原 PPO 参考观测最大误差约 `2.4e-7`；
从第一步 checkpoint 恢复后，第二步权重与连续训练逐项相等。
实验产物保留在 `outputs/dp_smoke_20260923/`，仅用于验证，不是训练完成的策略。
