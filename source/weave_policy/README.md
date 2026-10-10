# DINOv2 → reference Flow Matching → WEAVE PPO

这是不依赖 Isaac Sim / LeRobot / diffusers 的独立训练包。GR00T 对齐的是层级与计算公式；
WEAVE reference 的维度、图像条件和网络宽度按本项目协议替换。没有 state 或文本 token。

## 网络与数据协议

| 阶段 | 张量 / 实现 |
| --- | --- |
| 当前图像 | RGB `uint8[B,224,224,3]`，冻结官方 `dinov2_vitb14`，ImageNet normalization |
| 图像条件 | CLS + 全部 256 patch，`[B,257,768]`；可训练 LayerNorm |
| 参考目标 | `ref_robot[B,10,62]` + `ref_object[B,10,63]` → `[B,10,125]` |
| Flow Matching | `u~Beta(1.5,1)`，`t=(1-u)*0.999`，`x_t=(1-t)*noise+t*ref`，监督 `ref-noise` |
| 参考 encoder | category linear `125→768`，离散 time `floor(t*1000)` 广播 10 步，sin/cos `768` |
| time 融合 | concat `768+768=1536` → category linear `1536→768` → Swish → category linear `768→768` |
| 序列位置 | 10 个可训练位置 embedding，和参考 token 相加 |
| DiT | 16 blocks，16 heads × 48；偶数 block 图像 cross-attention，奇数 block reference self-attention |
| 每个 block | AdaLN(time) → Q/K/V attention → output projection/dropout → final dropout → residual；LayerNorm → FFN → residual |
| FFN | Linear `768→3072` → GELU(approximate=tanh) → dropout → Linear `3072→768` → dropout |
| 全局 time | 独立 256 维 cos/sin → Linear `256→768` → SiLU → Linear `768→768`；用于 AdaLN 和末尾 shift/scale |
| DiT 输出 | time shift/scale 调制 LayerNorm → Linear `768→768`，`[B,10,768]` |
| decoder | category linear `768→768` → ReLU → category linear `768→125`，预测 flow velocity |
| 推理 | 4 次 Euler，`t=0,0.25,0.5,0.75`，再反归一化、投影 rot6d、离散 contacts |

`125` 的字段顺序固定为 joint position 53、pelvis position 3、pelvis rot6d 6、object position 3、
object rot6d 6、contact 54。四元数为 `wxyz`；rot6d 是旋转矩阵前两列按行 flatten。
每个预测窗口的所有 pose 都相对**采样时刻实际 pelvis**，位置为直接坐标，不是增量。
实际 pelvis 只用于构造训练目标以及运行时坐标转换，不作为网络条件输入。

10 个 reference 时间为控制帧 offset `[0,5,10,15,20,25,30,35,40,45]`，默认 50 Hz。
WEAVE PPO 使用 `[0,5,10,15,20]`；默认 `--replan-steps 1`，直接取前 5 个 token。
更大的 replan 间隔使用位置/关节线性插值、rot6d 线性插值后投影、contact 最近邻，属于粗时间网格近似；
缓存轨迹始终保留预测时 pelvis anchor，然后转换到当前 pelvis frame。

归一化三种模式：

- `minmax`（默认）：训练集 `q01/q99` 映射至 `[-1,1]`，和 GR00T 默认 percentile min/max 一致。
- `minmax_extrema`：训练集真实 `min/max`。
- `meanstd`：训练集 mean/std。

默认 `clip_outliers=true`。contacts 固定采用 `[-1,1]`，保持 `-1/0/1` 含义；常量维归一化为 0。
序列末尾复制最后一个有效 target，mask 排除 padding loss。窗口不跨 episode。
训练/验证按 motion family 分组，`*_vN` 同源变体不会跨 split；统计只使用训练集有效 target。

## 文件结构

```text
weave_policy/
  config.py                  网络、时间与归一化配置
  data/
    reference_codec.py       125 维协议、坐标转换、WEAVE flatten、缓存
    reference_dataset.py     原始 HDF5 窗口、padding mask、图像/特征读取
    split.py                 防止 motion family 泄漏
    normalization.py         统计、三种归一化及反归一化
  models/
    dinov2_encoder.py        冻结图像编码器
    reference_encoder.py     GR00T action/time MLP
    timestep_encoder.py      DiT 全局 time MLP
    dit.py                   GR00T block 层级
    reference_decoder.py     768→768→125 velocity decoder
  flow_matching_policy.py    损失、Euler 推理、checkpoint
  predictor.py               RGB/token 推理及 encoder 一致性检查
  weave_adapter.py           WEAVE 协议检查、诊断 reference replay
```

## 安装

在已有 PyTorch + torchvision + h5py 的环境中安装，避免改动 Isaac 的 Torch 版本：

```bash
cd /data/g1_hoi_ws/Weave
python -m pip install --no-deps --no-build-isolation -e source/weave_policy
```

已有 DINOv2 checkout 可通过 `--dino-repo /data/g1_hoi_ws/dinov2` 使用；首次加载自动下载官方
ViT-B/14 权重。离线可再提供 `--dino-weights /path/to/dinov2_vitb14_pretrain.pth`。
图像缓存与 checkpoint 保存权重 SHA256、预处理参数和 encoder ID，推理拒绝不一致的 encoder。
原 `g1_hoi_learning.vision.dinov2_encoder` 是该实现的兼容导入路径。

## 从采集数据训练

下面命令从仓库根目录执行；输出路径必须尚不存在。默认完整网络，不传 `--smoke`。

```bash
python scripts/reference_policy/split_dataset.py \
  datasets/floorlamp_rollout_100_seed42/rollouts.h5 outputs/reference_split.json

python scripts/reference_policy/compute_stats.py \
  datasets/floorlamp_rollout_100_seed42/rollouts.h5 outputs/reference_stats.json \
  --split outputs/reference_split.json

python scripts/reference_policy/encode_images.py \
  datasets/floorlamp_rollout_100_seed42/rollouts.h5 outputs/reference_dino.h5 \
  --device cuda:0 --batch-size 8 --dino-repo /data/g1_hoi_ws/dinov2

python scripts/reference_policy/train.py \
  --source datasets/floorlamp_rollout_100_seed42/rollouts.h5 \
  --split outputs/reference_split.json --stats outputs/reference_stats.json \
  --features outputs/reference_dino.h5 --output outputs/reference_train \
  --device cuda:0 --amp --batch-size 32 --steps 100000
```

不传 `--features` 时，每次读取当前 RGB，由冻结 DINO 编码，需传 `--dino-repo`。
缓存使用 float16，不包含 robot state。100 条、36,365 帧数据的全部 token 未压缩约 13.4 GiB。
`--max-frames` 仅适用于检查编码/单帧推理，训练会检查所选 episode 的缓存覆盖率。

优化器采用 GR00T Adam `lr=1e-4, betas=(0.95,0.999), eps=1e-8, weight_decay=1e-6`；
500 步 warmup 后 cosine，gradient norm clip 1。`--config` 接受 `ReferencePolicyConfig` 的 JSON 字段。
例如 `{"normalization":"meanstd"}`，未提供的字段仍采用默认配置。

checkpoint 包含 config、训练统计、模型、encoder metadata、数据和相机 metadata，以及 optimizer、
scheduler、step 和 RNG。恢复需维持相同训练预算、batch、seed、配置和数据，输出到新目录：

```bash
python scripts/reference_policy/train.py \
  --source datasets/floorlamp_rollout_100_seed42/rollouts.h5 \
  --split outputs/reference_split.json --stats outputs/reference_stats.json \
  --features outputs/reference_dino.h5 --device cuda:0 --amp --batch-size 32 --steps 100000 \
  --resume outputs/reference_train/step_00010000 --output outputs/reference_resume
```

## 离线推理与 WEAVE 接入

```bash
python scripts/reference_policy/infer.py \
  --checkpoint outputs/reference_train/step_00100000 \
  --feature-cache outputs/reference_dino.h5 --row 0 \
  --device cuda:0 --output outputs/reference_prediction.npz

# 或使用原始 HDF5 的当前 RGB；也可 --rgb /path/to/image.npy
python scripts/reference_policy/infer.py \
  --checkpoint outputs/reference_train/step_00100000 \
  --source datasets/floorlamp_rollout_100_seed42/rollouts.h5 --episode 0 --frame 0 \
  --dino-repo /data/g1_hoi_ws/dinov2 --device cuda:0 --output outputs/reference_rgb_prediction.npz
```

输出 NPZ 包含物理量 `reference[B,10,125]`、`ref_robot[B,10,62]`、`ref_object[B,10,63]`、offsets 和 fps。
decoder 本身输出的是 velocity；Euler 积分、反归一化之后才是这些物理参考轨迹。

在 Isaac 环境中启动：

本机 `sim51` 若尚未配置 Isaac 环境变量，可把下面命令中的 `python` 换为
`PYTHONEXE=/data/g1_hoi_ws/sim51/bin/python /data/g1_hoi_ws/isaacsim/python.sh`。

```bash
python scripts/rsl_rl/play_reference.py --headless --device cuda:0 \
  --checkpoint /path/to/original_weave/model.pt \
  --reference-checkpoint outputs/reference_train/step_00100000 \
  --reference-file /path/to/initialization_motion.npz --clip-id 0 \
  --dino-repo /data/g1_hoi_ws/dinov2 --replan-steps 1 \
  --output outputs/reference_play
```

`--checkpoint` 是原 WEAVE PPO 权重，默认从其邻近 `params/agent.yaml` 恢复 agent 配置。
`--reference-checkpoint` 是新 flow 模型。初始化 NPZ 用来创建场景和 robot 初始姿态。
相机使用训练数据记录的 calibration；缺少 metadata 时显式提供采集的 `--camera-config resolved_config.json`。
程序在 PPO observation normalization 之前替换 `ref_motion_body/ref_motion_object`，保留实时 PPO
本体感知与物体观测。flow 模型只接收图像；PPO 最终仍输出 41 维机器人动作。

`--reference-source motion` 是诊断模式，会逐步对照原 PPO reference observation，绕过 flow 预测；
默认 `policy` 使用实时 DINO + flow。运行配置只在这个入口禁用依赖原动作参考的 termination、
固定初始化 clip、不做周期 motion reset，保留 timeout 和独立跌倒/非有限状态停止；不是任务成功指标。

## 验证

```bash
PYTHONPATH=source/weave_policy python -m pytest -q \
  source/weave_policy/tests source/g1_hoi_learning/tests/test_dinov2_encoder.py
```

数值对比测试默认读取 `/home/whynot/Isaac-GR00T`，可通过 `GR00T_SOURCE` 设置其他路径；需要 diffusers。
其余测试无需下载图像模型、启动 Isaac 或安装 diffusers。CLI 测试使用测试专用 token fixture，
验证训练、checkpoint 恢复和推理，不作为预训练图像编码器的效果验证。
`--smoke` 仅使用 2 blocks、96 维隐藏层，供流程检查，不能作为完整 768 维模型的训练结果。

本次 CPU 验证产物位于仓库 `outputs/reference_impl_validation/`：真实数据 split/统计、2 帧官方
DINO token、2 步 smoke checkpoint、推理 NPZ，以及完整默认模型的前向/反向/4 步积分检查记录。
真实 reference 转换与采集的 PPO observation 最大误差低于 `6e-7`。
当前机器 NVIDIA NVML 报 `NVML_ERROR_LIB_RM_VERSION_MISMATCH`，Torch/PhysX CUDA 报 error 804；
Isaac 入口已执行到环境创建，但无法完成 GPU 仿真闭环。以上 smoke 产物不代表策略已训练收敛。
