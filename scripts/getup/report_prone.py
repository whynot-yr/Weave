"""Summarize prone exploration, complete state exports and PhysX contacts."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/getup_prone_refined'));a=p.parse_args();out=a.output;s=json.loads((out/'results.json').read_text());entries=s['candidates'];best=next(x for x in entries if x['label']==s['selected'])
for e in entries:
 with (out/(e['label']+'_joint_state.csv')).open('w') as f:
  w=csv.writer(f);w.writerow(['joint_name','active','position_rad','velocity_rad_s'])
  for n,q in zip(s['names'],e['joint_pos_53']):w.writerow([n,n in s['drive_names'],q,0.])
with (out/'geometric_contacts.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['candidate','body','shape','nominal_lowest_x','nominal_lowest_y','nominal_lowest_z','random_near_ground_count'])
 for e in entries:
  for sh,pt,count in zip(s['shapes'],e['nominal_lowest_points'],e['near_ground_count_per_shape']):w.writerow([e['label'],sh['body'],sh['path'],*pt,count])
fig,ax=plt.subplots(1,2,figsize=(13,5));labels=[e['label'] for e in entries];ax[0].bar(labels,[e['valid_no_penetration']/1024*100 for e in entries]);ax[0].set_ylabel('No ground penetration (%)');ax[0].set_ylim(0,105);ax[1].bar(labels,[1000*e['random_min_z'] for e in entries]);ax[1].axhline(0,color='black');ax[1].set_ylabel('Worst sampled minimum z (mm)')
for x in ax:x.tick_params(axis='x',rotation=40)
fig.tight_layout();fig.savefig(out/'candidate_comparison.png',dpi=160);plt.close(fig)
lines=['# 俯卧 Seed 探索与验证','',f"优选候选：{s['selected']}。当前正式仰卧 reset 未修改。本次目的是提供可审阅的俯卧候选，而非自动切换训练初态。",'',
'## 方法与搜索过程','',
'参考 HoST g1_ground_prone 的 default_joint_angles，并将其 xyzw=[0,1,0,1] 方向归一化为当前 wxyz 四元数。HoST 的 root=0.5 m 是离地初始化，未复用其高度。参考姿态和仰卧翻转对照都按当前胸腹/骨盆前侧几何贴地，因此不能将参考候选的穿透率解释为 HoST 自身离地 reset 的穿透率。','',
'使用当前 WEAVE G1 + Inspire Hands 导入 USD 的 68 个碰撞形状。head_link 虽合并到 torso 刚体，其 USD 形状路径仍可单独识别，本次直接检查该形状，未用虚拟 head point 代替碰撞体。Root 高度由胸腹/骨盆前侧最低点加 1.5 mm 间隙计算，未逐样本抬高。搜索中 root 约束范围 0.04–0.12 m，nominal 胸部最低点不高于 8 mm，头部最低点至少 15 mm。','',
'首轮局部优化找到抬臂姿态，1024 验证通过，但手部离地约 21.8 cm。随后尝试限制抬臂和调整肩 yaw，部分解出现双手向背部中间靠拢，未作为最终推荐。进一步随机搜索 12000 组对称手臂参数，筛选手腕分别位于身体左右两侧的候选，选取最多 500 个做 256 个扰动拟合测试；三组较低手臂候选在独立 1024 验证中仍有失败。最后以其中一组作局部鲁棒优化，使用旧验证样本及 192 个扰动范围端点组合拟合，再使用新 seed=2026110105 进行最终独立比较。','',
'全过程保留 41 个主动关节独立随机化 q_seed*U(0.9,1.1)+U(-0.1,0.1)，先按兼容 mimic 的限位裁剪，再生成 12 个从动关节。Root 固定，初始线速度、角速度及全部关节速度为零。未通过缩小随机范围、拒绝失败样本或逐样本抬升 root 提高通过率。','',
'最终的 1024 样本表未用于最后一次拟合，所有下表候选使用同一张表。不同轮次使用不同 seed，统计数值以本目录为最终比较依据。','',
'## 最终 1024 个静态 reset 状态比较','', '| 候选 | Root 高度 m | nominal 最低点 mm | 无穿透 | 容许1mm时通过 | 最差样本最低点 mm | nominal 手部最低点 cm |','|---|---:|---:|---:|---:|---:|---:|']
for e in entries:lines.append(f"|{e['label']}|{e['root_pos'][2]:.6f}|{e['nominal_min_z']*1000:.3f}|{e['valid_no_penetration']}/1024|{e['valid_1mm_tolerance']}/1024|{e['random_min_z']*1000:.3f}|{e['nominal_hand_min_z']*100:.2f}|")
lines+=['',f"优选候选关节限位最大越界 {best['joint_limit_violation']} rad，mimic 误差 {best['mimic_error']} rad。1024 个样本中，前侧支撑形状近地（最低点在 [-1,5] mm）样本数 {best['random_front_near_ground']}；头部形状最低高度的最差值 {best['random_head_min_z']*100:.2f} cm。几何最低点及每个形状的近地次数见 geometric_contacts.csv。",'',
'以上是静态 reset 状态的几何筛查；并非 1024 个样本均进行了物理 rollout。68 个 USD 形状的检查不包括最终 PhysX cooking 容差、contact/rest offset，也不等于所有连续扰动组合的数学保证。','',
'## 优选姿态','',f"Root position = {best['root_pos']} m",'',f"Root quaternion (wxyz) = {best['root_quat_wxyz']}",'', '| 关节 | 左 rad | 右 rad |','|---|---:|---:|']
for part in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll','shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']:
 lines.append(f"|{part}|{best['joint_pos']['left_'+part+'_joint']:.6f}|{best['joint_pos']['right_'+part+'_joint']:.6f}|")
lines+=['',f"Waist pitch = {best['joint_pos']['waist_pitch_joint']:.6f} rad；waist roll/yaw = 0。手指主动 seed 为零，从动关节由 mimic 计算。完整 41D 主动角、53D 关节位置、53D 初始零速度及 13D root 状态均在 {s['selected']}.json 中，数组关节顺序见 results.json。",'',
f"该姿态是前侧支撑、躯干和腿部略抬起的俯卧姿态，不是四肢完全平贴地的放松姿态。nominal 胸部最低点 {best['nominal_chest_min_z']*1000:.2f} mm，头部最低点 {best['nominal_head_min_z']*100:.2f} cm，手部最低点 {best['nominal_hand_min_z']*100:.2f} cm。手臂比首轮抬臂候选低，但未达到手掌贴地。左右手腕位置分别为 {best.get('left_wrist_position')}、{best.get('right_wrist_position')}。",'',
'## PhysX：保持 seed 与零动作对照','']
if (out/'dynamics.csv').exists():
 rows=list(csv.DictReader((out/'dynamics.csv').open()));groups={}
 for r in rows:groups.setdefault(r['case'],[]).append(r)
 lines+=['16 个并行环境：HoST 参考、仰卧翻转、优选 nominal 和五个最差扰动样本，各用保持 seed / 零动作两种方式运行 1 秒（50 控制步）。保持 seed 的 action=q_seed-q_default；零动作对应 WEAVE 默认站姿目标。对照组含已知初始穿透，仅用于诊断，未作为训练候选。','', '| case | 最大base速度 m/s | 最大关节速度 rad/s | 最大单刚体净力 N | 头部几何最低点 mm | 终止次数 |','|---|---:|---:|---:|---:|---:|']
 for rs in groups.values():lines.append(f"|{rs[0]['label']}|{max(float(r['base_speed']) for r in rs):.3f}|{max(float(r['joint_speed']) for r in rs):.3f}|{max(float(r['contact_force']) for r in rs):.1f}|{min(float(r['head_geometry_min_z']) for r in rs)*1000:.3f}|{sum(r['terminated']=='True' for r in rs)}|")
 c=np.load(out/'contact_forces.npz');stats=[]
 for i,label in enumerate(c['labels']):
  for j,body in enumerate(c['body_names']):
   f=c['forces'][:,i,j];stats.append({'case':str(label),'body':str(body),'peak_net_force_N':float(f.max()),'control_steps_above_1N':int((f>1).sum()),'last_step_force_N':float(f[-1])})
 with (out/'contact_stats.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=stats[0]);w.writeheader();w.writerows(stats)
 lines+=['','优选 nominal 的实际前侧相关刚体接触：','']
 for r in stats:
  if r['case'] in ('hold_'+s['selected'],'zero_'+s['selected']) and r['body'] in ('pelvis','torso_link'):lines.append(f"- {r['case']} / {r['body']}: 峰值 {r['peak_net_force_N']:.1f} N，超过 1 N 的控制步 {r['control_steps_above_1N']}/50，末步 {r['last_step_force_N']:.1f} N。")
lines+=['','优选候选的 elbow seed 接近关节上限，扰动后存在较多限位裁剪；selected_clipping.csv 保存原始采样越过上下限的真实次数。随机化公式及原有裁剪规则未改变，但最终扰动分布会受基准角接近限位的影响。', '',
'保持 seed 与零动作两组中，头部几何都可能在落地后接近或轻微低于地面；零动作下优选及其扰动样本的峰值关节速度约 20–22.5 rad/s，高于首轮抬臂候选。手臂姿态更合理、静态通过率达标，不代表动力学方面全面优于其他候选。尤其是肘部从较大屈曲角快速回到默认站姿目标，会造成明显初始动作。此次未改变动作 offset 或增加过渡控制。', '',
'接触传感器提供刚体净力，不是逐形状接触点力。头部形状与 torso 合并为同一刚体，无法把 torso 净力直接归因为头部或胸部；动态头部几何接近/低于地面需单独关注，不能据此断言头部没有承重。静态合法不等于零动作下保持原姿态或始终无碰撞，未验证策略起身成功。沿用关闭自碰撞设置，左右手腕分离和图片检查不构成完整自碰撞保证。','',
'## 图片、数据和目录','',f"[优选侧视]({s['selected']}_side.png) · [优选斜视]({s['selected']}_oblique.png) · [优选俯视]({s['selected']}_top.png) · [保持目标1秒](hold_{s['selected']}_after_1s.png) · [零动作1秒](zero_{s['selected']}_after_1s.png) · [比较图](candidate_comparison.png)",'',
'本目录为最终独立比较。getup_prone_exploration 保存首轮搜索、抬臂候选和全局手臂搜索；getup_prone_arms_exploration / getup_prone_arms_yaw_exploration 保存中间尝试；getup_prone_final 是最后手臂细化前的验证集，已用于后续优化，不能当作最终独立验证。各目录保留原始结果，未覆盖。','',
'每个候选 json 和 *_joint_state.csv 保存完整状态；npz 保存 1024 个姿态、68 个形状最低点及每个样本的全身最低支撑点；randomness.npz 保存共享随机数；dynamics_initial.npz 和 dynamics_trajectory.npz 保存真实物理初态与轨迹；contact_stats.csv 保存逐刚体接触统计。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['source/g1_hoi_learning/g1_hoi_learning/tasks/getup/pose_cfg.py','source/g1_hoi_learning/g1_hoi_learning/tasks/getup/mdp/reset.py','source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf']
(out/'provenance.json').write_text(json.dumps({'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'validation_seed':s['validation_seed']},indent=2))
print('Report:',out/'REPORT.md')
