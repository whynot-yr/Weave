# g1_hoi_learning 交接文档

> 写给一个完全没有上下文的新对话。读完这份 + 项目根 `CLAUDE.md` + 记忆目录 `MEMORY.md`,应该能无缝接手。
> 最后更新:2026-07-18。当前分支 `develop`,工作区干净(latest commit `76eda83`)。

---

## 0. 工作方式(必须遵守)

- **对话用中文,代码 + 注释用英文,注释尽量少。**
- **改代码前先给方案、等确认再动**("settle form before editing")。不要一上来就 Edit。
- 四元数一律 **wxyz**。`numpy<2.0`。Python 3.11,venv 在 `../sim51`。
- 每个子项目是独立 git repo。

---

## 1. 任务是什么

`g1_hoi_learning` 是一个 Isaac Lab 扩展:让 **Unitree G1(29-DOF 身体 + Inspire 灵巧手)** 通过 RL 模仿"人-物交互(HOI)"参考动作 —— 机器人一边跟踪自身参考运动,一边操作一个物体(桌子/箱子/落地灯/椅子等),手上还带**接触标签(contact-label)监督**。

参考动作和物体从每条 motion 的 `.npz` 文件加载(`./data/train/*.npz`),多物体 round-robin 分配到并行环境一起训。

**两个已注册任务:**
| 任务 ID | 用途 | config 目录 |
|---|---|---|
| `G1-Inspire-HOI-v0` | tracker(特权观测,模仿参考动作) | `./configs/track` |
| `G1-Inspire-HOI-Distill-v0` | distill+RL(把 tracker 蒸馏成 goal-conditioned 盲策略) | `./configs/distill` |

**用户认定的核心难点(原话):** "最难的还是在于怎么学会抓握"。所有奖励/观测/编码器的努力都是围绕"让策略学会稳定抓握并搬动物体"。

---

## 2. 已经完成了什么

按最近几个 commit 展开(`git log --oneline` 从新到旧):

### `76eda83` lower minimum learning rate for simba v2
- 把 adaptive-KL 学习率下限从 `1e-5` 降到 `1e-6`。**⚠️ 只改了 `ppo/algorithm.py:102`,没改 `distillation/algorithm.py:117`(那份还是 `1e-5`)—— 见第 3 节卡点。**
- 背景:之前诊断 prepend run 时发现 LR **74% 的 iteration 贴在 1e-5 地板上**,adaptive-KL 想往下走被挡住,所以降 floor 对症。

### `3ead6dc` simba v2; lower learning rate; hand pose reward in object frame
- **SimBa → SimBaV2 改造(重点,影响最大)**:`algorithms/networks.py` 里的 `SimBa` 类整个换成超球面归一化版本(arXiv:2502.15280),参考 RL-X 的 flax 实现。新模块:`l2normalize / Scaler / HyperDense / HyperEmbedder / HyperMLP / HyperLERPBlock`。去掉了 LayerNorm、旧 SimBaBlock、`__getitem__`、reward scaling。**用户明确要求"不要做兼容保护"—— 这是硬替换,老 SimBa checkpoint 全部作废。**
  - 超参:`scaler_init=scaler_scale=sqrt(2/hidden)`,`alpha_init=1/(num_blocks+1)`,`alpha_scale=1/sqrt(hidden)`,`c_shift=3.0`,output head 是 `HyperDense→Scaler→Linear(orthogonal gain 0.01)`。
  - 签名:`SimBa(input_dim, output_dim, hidden_dim, num_blocks, expansion=4, c_shift=3.0)`。
- **加回 hand-in-object-frame 奖励** `motion_hand_obj_relative_pos_error_exp`(`rewards.py`):手部 body 在物体局部系里的位置追踪误差,exp 形式。cfg 里叫 `hand_obj_rel_pos`(weight 2.0, std 0.1,手指 intermediate/distal bodies)。**注意:这个函数之前被误删过一次,函数本体 + RewTerm + yaml 三处都得在。**

### `008f309` readme + `a99ca53` point cloud encoder / init randomization / feet slip
- **冻结的 PointNet++ 物体点云编码器**:用 `ssh point` 上预训练的 **seg** 模型(用户朋友建议 seg 比 cls 好)。权重在 `source/g1_hoi_learning/g1_hoi_learning/models/weights/pointnet2_rl_seg.pth`。封装在 `models/object_encoder.py`(`ObjectPointCloudEncoder` / `get_object_encoder`),接进 `object_point_cloud_b` 观测。输入点云必须先 **center + 单位球归一化**(和训练时 `pca_object_test.load_and_sample` 一致)。`pointnet2_ops` 已在本地 4090 / A800 / H20 三台机器编好。
- **物体 reset 位置扰动**:`object_range: {x:[-0.05,0.05], y:[-0.05,0.05]}`(**只给 xy,不给 orientation**,因为物体重力轴不严格是 z,给 ori 扰动容易直接触发 termination)。
- **feet_slide 惩罚**:`rewards.py` 里的 `feet_slide`,接触地面时惩罚脚的水平速度。配了 `feet_contact_sensor`(ContactSensorCfg on `.*_ankle_roll_link`)。当前 weight `-1.0`。
- git 结构:新增 `third_party/`(vendored `pointnet2_ops_lib`)和 `models/`。
- README 美化:hero SVG(play 视频第 5 帧,G1 抓桌子)、badges、mermaid 图、emoji 标题。tagline 只留 "Contact-aware humanoid-object interaction"。

### `b03437c` distillation obs-group; teacher weight inherit(蒸馏重构)
- 去掉纯 DAgger,**只留 distill+RL**。
- 设计:teacher = tracker 的 actor(冻结);student critic = tracker 的 critic(全量加载,因为 critic 和 teacher 阶段观测一样);student actor 复用 tracker 的 `object_state`/`robot_proprio` 编码器 + SimBa backbone,**goal 编码器(goal_root / goal_object)新初始化**。
- 实现 `goal_root` 观测(参考已有的 `goal_object`)。distill env 有 **7 个 obs group**(goal_root, goal_object, ref_motion_body, ref_motion_object, object_state, robot_proprio, robot_privileged),**这 5 个 tracker 组是在 `distill/env_cfg.py` 里重新 inline 定义的**(用户要求,不从 hoi import,免得看不到组里是什么)—— 所以要和 `hoi/env_cfg.py` 的 `ObservationsCfg` **手动保持同步**。
- `SimBaActorCriticTeacher`(`distillation/networks.py`):3-way 部分加载。`load_state_dict` 返回 False(加载 tracker checkpoint,走部分初始化)/ True(加载含 `teacher.*` 的 distill checkpoint,全量 resume)。rsl_rl runner 用这个返回值 gate 优化器加载 + iter 恢复。

### 其它已完成
- `MuonPPO.update` 里加了 `clip_fraction` 和 `kl` 日志(`Loss/clip_fraction`、`Loss/kl`,和 `Loss/learning_rate` 同组)。诊断脚本 `scratchpad/klcheck.py`。
- 彻底移除了 adaptive RSI sampling(见坑 #1)。

---

## 3. 当前卡在哪里 / 未决项

1. ~~**LR floor 两文件不一致。**~~ **已解决(2026-07-18,选 A 硬编码):`distillation/algorithm.py:117` 已同步成 `max(1e-6, ...)`,与 `ppo/algorithm.py:102` 一致。** 两份 update 仍是**独立复制的一份**,以后改 LR 逻辑要记得同步两处(见坑 #7)。

2. **SimBaV2 让所有旧 checkpoint 作废。** 必须**先从头重训 tracker(V2)**,再拿新 tracker 去 distill。不能把旧 SimBa tracker 加载进 V2。

3. ~~**distill 的 encoder 维度必须匹配 V2 tracker checkpoint。**~~ **已解决(2026-07-18):distill 的整套网络架构已对齐到 `configs/track/train.yaml` 的有效 tracker —— `hidden 1024 / expansion 4 / latent 128`,actor+teacher `num_blocks 1`、critic `num_blocks 2`,`encoder_hidden_dims` 五个共享组 = `[512,256]/[256]`,goal_root/goal_object 新增 `[256]`。改动落在 `configs/distill/train.yaml`(显式列出 latent_dim+encoder_hidden_dims)+ `rsl_rl_distill_cfg.py`(base 默认同步,消除过时值)。顺带对齐:distill 初始 `learning_rate 1e-3→1e-4`、`feet_slide -0.1→-1.0`、补齐显式 `hand_obj_rel_pos` 奖励(有效值本就一致,继承自 hoi base)。**⚠️ 前提仍是先重训 V2 tracker(见 #2),且 tracker 若再改 `configs/track/train.yaml` 的网络维度,distill 这套要跟着同步(现在是手动对齐,不是自动继承)。**

4. **SimBaV2 + 24G 显存 OOM。** 根因:`expansion=4` × minibatch 32768 → 8192 维中间张量 ≈ 1GiB。缓解:`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`expansion` 4→2、`num_mini_batches` 4→8、或 `hidden_dim` 2048→1024。当前 `configs/track/train.yaml` 用的是 `hidden_dim 1024 / num_blocks(actor 1, critic 2) / expansion 4 / latent_dim 128`。

5. **Muon × 超球面归一化的交互未验证。** 首个 V2 run 建议考虑 `schedule: fixed` + `learning_rate: 1e-4`(RL-X 用固定 1e-4→5e-5,不是 adaptive-KL),甚至临时去掉 Muon 做干净对照。当前 yaml 还是 `schedule: adaptive, learning_rate: 1e-4, desired_kl: 0.01`。

---

## 4. 下一步计划

1. ~~决定并统一 LR floor~~ **已完成(选 A)** ;~~同步 distill 网络维度~~ **已完成(distill 已对齐到当前 track/train.yaml)**。见第 3 节 #1/#3。
2. **从头重训 V2 tracker**(`G1-Inspire-HOI-v0`),盯 `Loss/kl`(健康约 0.01–0.02)、`Loss/clip_fraction`、以及各 `Metrics/motion/error_*` 和 object 相关指标。先确认 V2 + 当前奖励组合能收敛、能抓握。
3. 确认不 OOM(必要时按卡点 #4 调 expansion/minibatch)。
4. tracker 训好后**直接**跑 distill(`G1-Inspire-HOI-Distill-v0`,`--load_run` 指向 V2 tracker run、`--checkpoint model_XXXX.pt`)—— 架构已预先对齐,不用再手改维度;**除非**期间又改了 `configs/track/train.yaml` 的网络维度,那就得把 distill 那套(yaml + `rsl_rl_distill_cfg.py`)手动同步过去。
5. 围绕"抓握"继续:关注 `hand_obj_rel_pos`(手在物体系里的相对位姿)和 `contact` 奖励曲线。

---

## 5. 绝对不要再踩的坑

1. **不要重新加 adaptive RSI sampling。** 已证明净有害(追着失败帧采样 → 饿死抓握阶段)。用户彻底删掉了,别"好心"加回来。
2. **`pointnet2_ops` 编译:torch 的 CUDA 大版本必须和 nvcc 匹配。** pip 的 `nvidia-cuda-nvcc-cu12` wheel **不完整(只有 ptxas)**,别用它;用 apt 装 `cuda-nvcc-* cuda-cudart-dev-* cuda-cccl-*`。
3. **验证 pointnet2_ops 时不要在 `third_party/pointnet2_ops_lib` 源码目录里跑。** cwd 会 shadow,import 到没有 `_ext.so` 的本地源码 → 触发 JIT fallback → 在 `compute_37` 上失败报 "OutOfMemoryError" / "ModuleNotFoundError: _ext"。**先 `cd` 出源码目录**再验证。编译产物是 site-packages 里的 ~27MB `_ext.so`。
4. **`motion_hand_obj_relative_pos_error_exp` 这个奖励:函数本体、`RewardsCfg` 的 RewTerm、yaml 三处必须同时在。** 之前只加 cfg 没加函数 → 运行时 AttributeError。
5. **SimBaV2 是硬替换,旧 SimBa checkpoint 全废。** 别尝试加载旧权重进 V2;改 V2 后 tracker 和 distill 都要重训。
6. **distill 的 5 个 tracker obs-group 是在 `distill/env_cfg.py` 里 inline 复制的**,不是 import 的 —— 改 `hoi/env_cfg.py` 的观测时**要手动同步过去**,否则 teacher/critic 权重加载会错位。
7. **改了 `ppo/algorithm.py` 的 `update` 逻辑,记得 distill 的 `update` 是独立复制的一份**,大概率要同步改(LR floor 就是活例子)。
8. **README:** emoji 标题不要用带 U+FE0F 变体选择符的(🗂️⚙️▶️ 会让锚点多出杂字符 `#️-...`),用干净 emoji(✨🧭📁🔧🧩🚀🎬🧹);mermaid 里不要用 `·`、`()`、`<br/>`,用 `graph LR` + 纯 ASCII。
9. **物体扰动只给 xy,不要给 orientation**(重力轴不严格是 z,ori 扰动会直接触发 termination)。
10. **distill 的 goal `gap` 上界:想让"目标到动作终点"要把上界设 `null`(=clip 剩余长度),不要塞大数。** `_sample_goal` 有 per-clip `clamp(max=clip_len-1)`,塞越大的上界越把 goal 死死夹到终点(旧 `gap:[50,1000]` 下 ~91% 的 goal 已堆在终点,上界形同虚设)。`gap[1]=None` 走 `U[ref+gap0, clip_len-1]` 逐 env 均匀采样,goal 才真正铺开近-中-远(终点占比 ~22%),配合 reach-resample 形成滚动 horizon。单条 clip 只有 ~150-750 帧(命名里的数字是 clip *条数* 不是帧数)。

---

## 6. 关键文件地图

```
configs/track/{train,play}.yaml          # tracker 的 hydra override(奖励权重/std、扰动范围、agent 网络与算法超参)
configs/distill/{train,play}.yaml        # distill 的 hydra override
source/g1_hoi_learning/g1_hoi_learning/
├── algorithms/
│   ├── networks.py                      # ★ SimBaV2 backbone(超球面归一化)
│   ├── optimizers.py                    # MuonAdamWWrapper(Muon on 2D 权重 + AdamW,ignore_frozen)
│   ├── ppo/algorithm.py                 # ★ MuonPPO.update(LR floor 在 line 102;clip_fraction/kl 日志)
│   ├── ppo/networks.py                  # SimBaActorCritic + build_group_backbone(GroupEncoder→SimBa)
│   └── distillation/
│       ├── algorithm.py                 # ★ MuonPPODistill.update(独立复制;LR floor 在 line 117,当前还是 1e-5)
│       ├── networks.py                  # SimBaActorCriticTeacher(3-way 部分加载)
│       └── __init__.py                  # on_policy_runner 注册(已去掉 DAgger)
├── models/
│   ├── object_encoder.py                # 冻结 PointNet++ seg 编码器封装
│   ├── pointnet2_rl_seg.py / _encoder.py
│   └── weights/pointnet2_rl_seg.pth
└── tasks/
    ├── hoi/                             # G1-Inspire-HOI-v0(tracker)
    │   ├── env_cfg.py                   # SceneCfg/ObservationsCfg/RewardsCfg/...
    │   └── mdp/{commands,observations,rewards,terminations}.py
    └── distill/                         # G1-Inspire-HOI-Distill-v0
        ├── env_cfg.py                   # 7 obs group(含 inline 复制的 5 个 tracker 组)
        └── mdp/{commands,observations}.py  # goal_root_* / goal_object_* / goal_root_pos_b 等
third_party/pointnet2_ops_lib/           # vendored(setup.py 里 TORCH_CUDA_ARCH_LIST)
logs/rsl_rl/g1_inspire_hoi/              # 训练日志 + checkpoint(exports JIT+ONNX)
```

记忆目录里还有更细的背景(`~/.claude/projects/.../memory/MEMORY.md` 索引),重点看:
`goal-conditioned-distill-plus-rl`、`pointnet2-frozen-object-encoder`、`prepend-training-diagnosis`、`communication-and-code-style`、`isaacsim51-mimic-joint-compliant`。

---

## 7. 常用命令

```bash
# 训 tracker
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 --headless \
    --config-dir ./configs/track --config-name train
# 回放 tracker
python scripts/rsl_rl/play.py  --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs/track --config-name play --load_run <RUN_DIR> --checkpoint <model_XXXX.pt>

# distill(--load_run 指向已训好的 V2 tracker run,其 actor 自动作为冻结 teacher)
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-Distill-v0 \
    --config-dir ./configs/distill --config-name train \
    --load_run <TRACKER_RUN_DIR> --checkpoint <model_XXXX.pt>

# OOM 时
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/rsl_rl/train.py ...
```
