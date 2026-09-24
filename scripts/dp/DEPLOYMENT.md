# DP → WEAVE PPO 部署

仅新增运行路径。原 `play.py`、PPO、MDP commands/observations 保持不变。
本实现参照 HumanoidArena 的 HTTP 推理、chunk 缓存和运行时参考适配结构；
不复制其根部增量积分、TWIST2 35D mimic 指令或独立夹爪控制。

## 环境与启动

两个进程各使用自己的环境。DP 服务只需要 checkpoint，不需要源 HDF5 或 Isaac Sim。
仿真端只安装轻量 `weave-data`，不要向 sim51 安装 LeRobot，也不要升级其 torch。

```bash
cd /data/g1_hoi_ws/Weave
uv pip install --python /data/g1_hoi_ws/sim51/bin/python --no-deps -e source/weave_data
```

终端一：

```bash
cd /data/g1_hoi_ws/Weave
.venv-dp/bin/python scripts/dp/serve.py \
  --checkpoint outputs/dp_train/step_00100000 \
  --device cuda:0 --host 127.0.0.1 --port 8765
```

CPU 冒烟测试使用 `--device cpu`；`.venv-dp` 必须有 CUDA 版 torch 才能使用 GPU。
单卡同时承载仿真、PPO、DP 可能显存不足，可以给 DP 分配另一 GPU，或先用 CPU 验证接口。

终端二（必须激活 sim51，以加载 Isaac Sim 自身的库路径）：

```bash
cd /data/g1_hoi_ws/Weave
source /data/g1_hoi_ws/sim51/bin/activate
python scripts/rsl_rl/play_dp.py \
  --task G1-Inspire-HOI-v0 \
  --checkpoint logs/rsl_rl/g1_inspire_hoi/2026-08-19_10-29-31_multi-obj-test/model_31400.pt \
  --reference_file data/test/floorlamp_33clip_var5.npz --clip-id 0 \
  --camera-config datasets/floorlamp_rollout_100_seed42/resolved_config.json \
  --dp-url http://127.0.0.1:8765 \
  --num_envs 1 --replan-steps 20 --episodes 1 --episode-seconds 20 \
  --max-steps 1000 --output outputs/dp_play_run01 --headless --device cuda:0
```

以上 checkpoint 路径是示例，DP 应换成正式训练的模型；不能把两步冒烟 checkpoint 当可用策略。
输出目录必须不存在。仅支持单物体初始化 NPZ、单环境、本机 HTTP。
PPO 默认读取 checkpoint 旁的 `params/agent.yaml` 恢复训练时网络结构、观测组及动作裁剪配置；
配置放在其他位置时使用 `--ppo-config /path/to/agent.yaml`。不能用当前任务默认网络结构猜测旧模型。
`reference_file` 保留原场景与状态初始化机制，不参与 DP 模式的目标参考生成。
`--camera-config` 必须来自对应训练数据的采集配置，以复用真实相机内外参而非猜测。
旧 checkpoint 元数据不含完整内外参，程序无法证明用户提供的采集配置就是训练时那一份。

## 单步数据流

1. 在同一仿真控制时刻读取 RGB 与实际机器人状态。
2. `encode_state` 构造初始 heading 校正后的 6D 姿态 + q41 + dq41。
3. 首步和每20步同步 POST `/infer`；等待期间不推进物理仿真。
4. 返回 `[40,125]`，在 WEAVE 端缓存该时刻的实际 pelvis 位姿和原始预测。
5. 第 j 步查询 `j+[0,5,10,15,20]`，j=0..19。
6. 将选中的5帧从预测时刻的 pelvis 坐标系转换到当前实际 pelvis 坐标系。
7. 投影 rot6d，按 ±0.5 阈值离散接触标签为 -1/0/1，按特征组展平。
8. 在 PPO 自带归一化之前替换 `ref_motion_body[1,310]` 和 `ref_motion_object[1,315]`。
9. 原 PPO 输出41维动作，经原 wrapper 的动作裁剪、原 action term 和 env.step 执行。

不对位置积分；不覆盖原始 chunk；不改 `object_state` 或 `robot_proprio`。
DP 不接收物体真值，但原 PPO 仍使用仿真实际物体状态。这不是实机纯视觉部署。

## HTTP 与状态管理

- `GET /health`：协议、模型、输入输出维度、fps、关节/身体顺序。
- `POST /reset`：绑定 session，开始递增的 episode，清理 policy/processor 状态，可指定 seed。
- `POST /infer`：RGB（uint8 原始字节 base64）、state88、episode/request/控制步编号。
  返回原单位参考、原编号和推理耗时。
- `POST /close`：释放 session，允许下一次运行使用服务。

一次只允许一个 session，推理串行；不同 session/重复请求/旧 episode 会被拒绝。
请求有体积上限，验证 RGB 字节数、shape、有限值和版本；不使用 pickle；不自动重试推理。
V1 强制 loopback，不能把它作为已加固的公网服务。服务不提供认证/TLS。
如果客户端异常退出导致 session 未释放，重启服务再运行。

reset 时清空 chunk、重置 episode heading、刷新相机。HTTP 错误、过期响应、非法预测、
PPO 非有限输入/动作均停止运行；不回退到原 motion、不继续执行过期 chunk。
模型请求超时默认30秒，可用 `--dp-timeout` 调整；这不是实时性承诺。

## 独立仿真配置

只对新入口的配置实例作调整：

- 固定单环境和初始化 clip；motion 使用 eval 模式且禁用定时重采样，结尾只截断、不重置机器人。
- 相机设置来自采集 resolved_config.json，224×224；仅渲染 RGB。
- 关闭启动/周期随机化及观测噪声，用固定 seed 评估。
- 移除依赖旧参考的 anchor/object/末端位置、接触丢失及 clip-end 终止。
- 保留独立时间限制；增加非有限实际状态、粗略跌倒终止。
- 跌倒默认 pelvis 高度 <0.25m 或实际机体重力方向 z >-0.2；高度可用 `--min-pelvis-height` 调整。

原 motion reward 和 metrics 仍可能被环境计算，但不作为 DP 成功率。当前不实现任务特定的成功判断。
粗略跌倒检测、旋转投影不是完整安全保障，正式闭环仍需逐步验证。

## 参考回放与测试

先在上面的仿真命令中加 `--reference-source motion` 并换一个新输出目录。
该模式不需要 DP 服务，使用原 motion 生成同样的40帧局部 chunk，经过缓存、取帧、重新定锚，
每一步与原 PPO 参考观测比较（容差2e-5）。这是明确选择的诊断模式，不能当 DP 效果测试。
推荐 `--max-steps 22` 跨越一次20步重规划边界。

```bash
OMP_NUM_THREADS=2 .venv-dp/bin/python -m pytest source/weave_data/tests -q
```

运行输出包含 `run_config.json` 和 `trace.jsonl`，记录模型标识、环境/相机设置、请求编号、
推理及往返耗时、动作最大绝对值、reset和终止原因；异常也记录在 trace 中。
目前不支持异步延迟补偿、多环境、自动任务成功率或实机控制。

## 已完成的本地冒烟验证

- 24项自动测试通过，包含 HTTP 会话隔离、超时、服务忙、过期请求、非法输出、
  参考缓存边界、坐标变换及现有训练数据链路回归测试。
- 原 PPO checkpoint + 真正 Isaac Sim 场景，参考回放22步（跨越20步重规划），
  共44次参考组对照，最大误差约 `4.8e-7`。
- 两步训练的小 DP checkpoint 通过独立 HTTP 服务与仿真/PPO 连接，执行22步，
  第0、20步成功请求预测；DP 使用 CPU，仿真/PPO 使用 GPU。
- 两个短 episode 共44步，每个 episode 均在第0、20步请求预测，时间终止后完成
  环境、相机、初始 heading、参考缓存及 DP session episode 状态重置，无接口错误。
  日志：`outputs/dp_deploy_reset_check/trace.jsonl`。
- 上述只验证接口、坐标变换和执行时序，不代表正式 DP 的任务成功率、长时稳定性或实时性能。
