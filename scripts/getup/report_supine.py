"""Report exploration data without changing the training seed."""
import json,csv,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
out=Path('outputs/getup_supine_exploration');s=json.loads((out/'fresh_results.json').read_text());entries=s['candidates'];best=entries[-1]
# Explicit schema and mapping audit, including added hand/wrist/waist joints.
hostpath=Path('/data/g1_hoi_ws/HoST/legged_gym/resources/robots/g1/g1_23dof.urdf');weavepath=Path('source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf')
hj={j.get('name'):j for j in ET.parse(hostpath).getroot().findall('joint') if j.get('type')!='fixed'};wj={j.get('name'):j for j in ET.parse(weavepath).getroot().findall('joint')};mapping=[]
for n,j in hj.items():
 k=wj[n];mapping.append({'name':n,'host_axis':j.find('axis').get('xyz'),'weave_axis':k.find('axis').get('xyz'),'same_axis':j.find('axis').get('xyz')==k.find('axis').get('xyz'),'host_origin':j.find('origin').attrib,'weave_origin':k.find('origin').attrib})
(out/'mapping.json').write_text(json.dumps({'mapped_23':mapping,'weave_active_joints_not_in_host':[n for n in s['drive_names'] if n not in hj]},indent=2))
# Export ordered complete state for each candidate with explicit joint names.
for e in entries:
 with (out/(e['label']+'_joint_state.csv')).open('w') as f:
  w=csv.writer(f);w.writerow(['joint_name','active','position_rad','velocity_rad_s'])
  for n,q in zip(s['names'],e['joint_pos_53']):w.writerow([n,n in s['drive_names'],q,0.])
fig,axes=plt.subplots(1,2,figsize=(13,5));labels=[e['label'] for e in entries]
axes[0].bar(labels,[e['valid_no_penetration']/1024*100 for e in entries]);axes[0].set_ylabel('No-penetration rate (%)');axes[0].set_ylim(0,105)
axes[1].bar(labels,[e['random_min_z']*1000 for e in entries]);axes[1].axhline(0,color='black');axes[1].set_ylabel('Worst sampled collision height (mm)')
for ax in axes:ax.tick_params(axis='x',rotation=40)
fig.tight_layout();fig.savefig(out/'candidate_comparison.png',dpi=160);plt.close(fig)
# Near-ground geometry is separate from measured force contacts.
with (out/'geometric_contacts.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['candidate','body','shape','nominal_lowest_x','nominal_lowest_y','nominal_lowest_z','random_near_ground_count'])
 for e in entries:
  for shape,point,count in zip(s['shapes'],e['nominal_lowest_points'],e['near_ground_count_per_shape']):w.writerow([e['label'],shape['body'],shape['path'],*point,count])
lines=['# 仰卧 Seed 探索与验证','',
'结论：refined 候选在新的独立验证集上达到 1024/1024 个完整随机初态无地面穿透，最小几何间隙约 1.06 mm。训练默认 reset 尚未替换。该结果是有限样本验证，不是整个连续扰动空间的数学保证。', '',
'## 参考、映射与搜索范围','',
'HoST G1 Ground 使用 default_joint_angles 初始化关节（hip pitch=-0.1、knee=0.3、ankle pitch=-0.2、elbow=0.8，其余参考角大多为零），root z=0.5 m，xyzw=[0,-1,0,1]；这是离地初始化。target_joint_angles 为另一套目标角，未将其误作 reset。这里将 root 四元数归一化为 wxyz，并按当前背部几何重新设置高度，没有复用 0.5 m。', '',
'两份 URDF 的 23 个共有活动关节轴方向相同，按名称映射，详见 mapping.json。当前模型的腰部机构、肩部安装位置、额外腕部自由度及 Inspire Hands 存在差异；新增关节以 WEAVE 默认值起步，之后允许选定关节局部优化。HoST 配置中的 waist pitch/roll 零值不代表其 23DOF 资产有这两个活动自由度。', '',
'搜索采用左右对称的 hip pitch、knee、ankle pitch、shoulder pitch/roll、elbow，以及 waist pitch 和 root pitch，共 8 个参数。Root pitch 搜索范围 [-1.72,-1.42] rad。每次由 pelvis/torso 后侧最低碰撞点计算 root 高度，使最低背部几何间隙为 1.5 mm；root 高度目标范围 0.045–0.10 m，nominal torso 最低点不高于 8 mm。高度不是可独立增大的优化变量。', '',
'第一轮三个局部起点各用 64 个扰动样本优化，独立 1024 样本验证后仍有失败。第二轮将这些失败所在的旧验证集纳入优化，并加入 128 个范围端点组合；以下表格使用新的 seed=20261101，共享 1024×41 的 scale/offset，所有候选均重新比较。新验证集没有用于第二轮优化。', '',
'随机化保持 q_seed*U(0.9,1.1)+U(-0.1,0.1)，使用兼容 mimic 的限位裁剪，再推导 12 个从动关节；root 姿态固定，初始所有速度为零。不进行逐样本抬高、投影或穿地重采样。', '',
'## 同设置 1024 次静态 Reset 状态检查','', '| 候选 | Root 高度 m | nominal 最低点 mm | 严格无穿透 | 容许 1mm 时通过 | 最差样本最低点 mm |','|---|---:|---:|---:|---:|---:|']
for e in entries:lines.append(f"|{e['label']}|{e['root_pos'][2]:.6f}|{e['nominal_min_z']*1000:.3f}|{e['valid_no_penetration']}/1024|{e['valid_1mm_tolerance']}/1024|{e['random_min_z']*1000:.3f}|")
lines+=['', '各候选关节限位最大越界和 mimic 最大误差均为 0。以上 1024 是静态 reset 状态几何筛查；物理 rollout 仅对下文列出的选定姿态执行。使用当前导入 USD 的 68 个碰撞形状，未用 visual mesh 代替 collider，未包含 PhysX cooking 容差和 contact/rest offset。','', '## 优选候选参数','',f"Root position = {best['root_pos']} m",'',f"Root quaternion (wxyz) = {best['root_quat_wxyz']}",'', '| 关节 | 左 | 右 |','|---|---:|---:|']
for part in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll','shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']:
 lines.append(f"|{part}|{best['joint_pos']['left_'+part+'_joint']:.6f}|{best['joint_pos']['right_'+part+'_joint']:.6f}|")
lines+=['',f"Waist pitch = {best['joint_pos']['waist_pitch_joint']:.6f} rad；waist roll/yaw = 0。手部主动关节沿用默认零值，mimic 从动位置按映射计算。完整有序 41D 主动角、53D 关节位置、53D 零速度及 13D root 状态见 refined_fresh.json；关节顺序在 fresh_results.json 的 names/drive_names 中。", '',
'姿态是面朝上的仰卧屈髋/屈腰姿态，主要几何支撑位于骨盆后侧；腰部约 22.6° 屈曲，肩臂轻度展开，躯干并非整片背部平贴地。nominal torso 最低点约 7.89 mm。该姿态是否符合最终期望的“平躺”程度，应结合图片判断。', '',
f"1024 个随机样本均有 pelvis/torso 的碰撞形状最低点位于 [-1,5] mm 的近地区间；其中 torso 近地的样本数为 {best['random_torso_near_ground']}。这是几何近地统计，不能等同于真实 PhysX 接触。逐形状最低支撑点 XYZ 与近地频次见 geometric_contacts.csv；每个样本的最低点 XYZ 在 *_fresh.npz 中。", '', '## PhysX 对照与接触','']
for filename,contactfile in [('dynamics.csv','contact_forces.npz'),('refined_dynamics.csv','refined_contacts.npz')]:
 if not (out/filename).exists():continue
 rows=list(csv.DictReader((out/filename).open()));groups={}
 for r in rows:groups.setdefault(r['case'],[]).append(r)
 lines+=['',f'数据：{filename}。正常推进 1 秒（50 控制步），保持 reset 关节目标，初速为零，未反复写入姿态。','', '| 姿态 | 最大 base 速度 m/s | 最大关节速度 rad/s | 最大单刚体净接触力 N | 终止次数 |','|---|---:|---:|---:|---:|']
 for rs in groups.values():lines.append(f"|{rs[0]['label']}|{max(float(r['base_speed']) for r in rs):.3f}|{max(float(r['joint_speed']) for r in rs):.3f}|{max(float(r['contact_force']) for r in rs):.1f}|{sum(r['terminated']=='True' for r in rs)}|")
 if (out/contactfile).exists():
  c=np.load(out/contactfile);forces=c['forces'];bodies=c['body_names'];labels=c['labels'];stats=[]
  for i,label in enumerate(labels):
   for j,body in enumerate(bodies):
    ff=forces[:,i,j];stats.append({'case':str(label),'body':str(body),'peak_net_force_N':float(ff.max()),'control_steps_above_1N':int((ff>1).sum()),'last_step_force_N':float(ff[-1])})
  with (out/(Path(contactfile).stem+'_stats.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=stats[0]);w.writeheader();w.writerows(stats)
  if contactfile=='refined_contacts.npz':
   lines+=['','优选 nominal 的背部相关实际接触（控制步内取子步净力峰值）：','']
   for r in stats:
    if r['case']=='refined' and r['body'] in ('pelvis','torso_link'):lines.append(f"- {r['body']}: 峰值 {r['peak_net_force_N']:.2f} N；50 步中 {r['control_steps_above_1N']} 步超过 1 N；末步 {r['last_step_force_N']:.2f} N。")
lines+=['','本检查验证保持 seed 目标的短时响应，不代表零策略动作（目标为默认站姿）下行为相同，也不证明可学会起身。资产沿用关闭自碰撞，未获得身体内部碰撞的完整保证。接触传感器给出刚体净力，不提供本报告中的精确接触点；几何支撑点与实际接触力分开报告。', '',
'## 图片与完整产物','', '[候选比较](candidate_comparison.png) · [优选侧视](refined_side.png) · [优选斜视](refined_oblique.png) · [优选俯视](refined_top.png) · [优选 1 秒后](refined_refined_run_after_1s.png)', '',
'每个候选的 *_fresh.json 保存 root、41D/53D 状态及统计；*_joint_state.csv 提供带关节名的完整状态；*_fresh.npz 保存 1024 个扰动姿态、68 个碰撞形状最低高度及最低点；fresh_randomness.npz 保存共享随机数。原始轮次保留在 results.json 和 shared_randomness.npz，最终比较以 fresh_results.json 为准。', '',
'复现入口：explore_supine.py → refine_supine.py（需要 Isaac Sim 的 pxr 库，建议 OPENBLAS_NUM_THREADS=1）→ validate_supine_candidate.py → report_supine.py。当前探索脚本使用记录的资产路径和输出目录，不应覆盖已有结果。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=[hostpath,weavepath,Path('/data/g1_hoi_ws/HoST/legged_gym/legged_gym/envs/g1/g1_config_ground.py'),Path('source/g1_hoi_learning/g1_hoi_learning/tasks/getup/pose_cfg.py'),Path('source/g1_hoi_learning/g1_hoi_learning/tasks/getup/mdp/reset.py')]
(out/'provenance.json').write_text(json.dumps({'sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}},indent=2))
print('Report written',out/'REPORT.md')
