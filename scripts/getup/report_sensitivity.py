"""Build sensitivity plots, paired ablation statistics and a readable report."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/getup_reset_sensitivity_074'));a=p.parse_args();out=a.output
s=json.loads((out/'results.json').read_text());rows=s['metrics']
full=np.load(out/'E_all.npz')['minima'].min(-1)
rest=[]
for r in rows:
    if r['kind']=='restore':
        z=np.load(out/(r['label']+'.npz'))['minima'].min(-1);diff=z-full
        rest.append({'joint':r['label'][8:],'mean_improvement_mm':float(diff.mean()*1000),'median_improvement_mm':float(np.median(diff)*1000),
                     'improved_gt1mm':int((diff>.001).sum()),'worsened_gt1mm':int((diff<-.001).sum()),
                     'rescued_minus1mm':int(((full<-.001)&(z>=-.001)).sum()),'remaining_below_minus1mm':int((z<-.001).sum())})
rest.sort(key=lambda x:x['mean_improvement_mm'],reverse=True)
with (out/'ablation.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rest[0]);w.writeheader();w.writerows(rest)
singles=sorted([r for r in rows if r['kind']=='single_random'],key=lambda x:x['min_z'])
fig,axes=plt.subplots(1,2,figsize=(15,8))
x=singles[:20];axes[0].barh([r['label'][14:] for r in x],[-r['min_z']*1000 for r in x]);axes[0].invert_yaxis();axes[0].set_xlabel('Depth below ground (mm; negative = clearance)');axes[0].set_title('Single-joint randomization: worst geometry')
x=rest[:16];axes[1].barh([r['joint'] for r in x],[r['mean_improvement_mm'] for r in x]);axes[1].invert_yaxis();axes[1].set_xlabel('Mean increase in global minimum height (mm)');axes[1].set_title('Restore one joint to default in full random poses')
fig.tight_layout();fig.savefig(out/'sensitivity.png',dpi=160);plt.close(fig)
baseline=next(r for r in rows if r['kind']=='baseline');e=next(r for r in rows if r['label']=='E_all')
lines=['# Root 0.074 m reset 敏感性测试','',
'此轮仅在诊断环境将 root 设置为 0.074 m；训练默认参数未修改。使用实际 RL reset 生成共享的 1024 个随机姿态，再按测试组保留指定主动关节的扰动，其余恢复默认并重新计算 mimic。没有改变随机化分布或裁剪规则。',
'',f"逐关节随机测试 41 组，每组 1024 个样本；逐关节扫描 41 组，每组 101 点；组合测试 21 组；完整随机姿态的逐关节恢复消融 41 组。另有 baseline 与全随机对照，共 {len(rows)} 个测试条件。",
'',f"几何依据为导入 USD 的 {s['shape_count']} 个碰撞形状。新几何实现对旧的 1024 个样本逐形状回归，最大误差 {s['geometry_regression_max_error']:.3g} m。固定 root、零初始速度、静态测量；不含 PhysX cooking 容差及 contact/rest offset，不检测自碰撞。",
'', '## 基础与完整随机化','', '| 条件 | 样本数 | 最低 z / mm | 中位 z / mm | z<0 数量 | z<-1mm 数量 |','|---|---:|---:|---:|---:|---:|']
for r in [baseline,e]:lines.append(f"|{r['label']}|{r['n']}|{r['min_z']*1000:.3f}|{r['median_z']*1000:.3f}|{r['below_0']}|{r['below_minus_1mm']}|")
lines+=['','## 单关节随机化排名','', '| 主动关节 | 最低 z / mm | z<-1mm 数量 /1024 | 最低碰撞体 |','|---|---:|---:|---|']
for r in singles[:15]:lines.append(f"|{r['label'][14:]}|{r['min_z']*1000:.3f}|{r['below_minus_1mm']}|{r['worst_body']}|")
lines+=['','## 单关节扫描方向','', '其他主动关节固定时，以下为 101 点扫描中观察到的危险范围；边界只有扫描分辨率精度，不能直接当作新的训练限位。waist_yaw 的范围可能包含多个不连续区间，表中仅给最小/最大值。', '', '| 关节 | default / rad | 最差角度 / rad | z<-1mm 的扫描角度范围 / rad |','|---|---:|---:|---|']
inputs=np.load(out/'inputs.npz')
fig,axes=plt.subplots(2,3,figsize=(13,7))
for ax,name in zip(axes.flat,['left_hip_pitch_joint','right_hip_pitch_joint','left_knee_joint','right_knee_joint','waist_pitch_joint','waist_yaw_joint']):
    d=np.load(out/('scan_'+name+'.npz'));j=s['names'].index(name)
    ax.plot(d['q'][:,j],d['minima'].min(-1)*1000)
    ax.axhline(-1,color='red',linestyle='--');ax.axvline(inputs['nominal'][j],color='grey',linestyle=':')
    ax.set_title(name);ax.set_xlabel('Joint angle (rad)');ax.set_ylabel('Global min z (mm)')
fig.tight_layout();fig.savefig(out/'scan_curves.png',dpi=160);plt.close(fig)
for r in sorted([x for x in rows if x['kind']=='scan'],key=lambda x:x['min_z']):
    name=r['label'][5:];j=s['names'].index(name);d=np.load(out/(r['label']+'.npz'));z=d['minima'].min(-1);angles=d['q'][:,j];bad=angles[z<-.001]
    if len(bad):lines.append(f"|{name}|{inputs['nominal'][j]:.6f}|{angles[z.argmin()]:.6f}|[{bad.min():.6f}, {bad.max():.6f}]|")
lines+=['','排名只描述其他主动关节固定时的影响。所有 41 个关节及扫描结果均见 metrics.csv。最低值是有限采样/扫描观察到的最坏值，不是解析全局极值。','', '## 组合测试','', '| 组 | 最低 z / mm | z<-1mm 数量 /1024 | 左脚最低 / mm | 右脚最低 / mm |','|---|---:|---:|---:|---:|']
for r in rows:
    if r['kind']=='group':lines.append(f"|{r['label']}|{r['min_z']*1000:.3f}|{r['below_minus_1mm']}|{r['left_foot_min']*1000:.3f}|{r['right_foot_min']*1000:.3f}|")
lines+=['','## 完整随机姿态中的恢复消融','', '| 恢复默认的关节 | 全身最低点平均改善 / mm | 改善>1mm 样本 | 恶化>1mm 样本 | 从穿地<-1mm恢复的样本 |','|---|---:|---:|---:|---:|']
for r in rest[:15]:lines.append(f"|{r['joint']}|{r['mean_improvement_mm']:.3f}|{r['improved_gt1mm']}|{r['worsened_gt1mm']}|{r['rescued_minus1mm']}|")
lines+=['','改善是配对计算 z_restore - z_full，正值表示减轻重叠。全身最低点会在左右脚或其他部位间切换，单关节恢复的改善量不能相加，也不等同于独立因果贡献百分比。完整配对统计见 ablation.csv。', '', '## 短时动力学','']
if (out/'dynamics.csv').exists():
    dr=list(csv.DictReader((out/'dynamics.csv').open()));groups={}
    for r in dr:groups.setdefault(r['case'],[]).append(r)
    lines+=['16 个选定姿态包含 baseline 和按静态重叠深度选择的不同最差姿态。正常推进 1 秒（50 控制步），固定采样后的关节目标；未运行策略。','', '| case / 来源 | 最大 base 速度 m/s | 最大关节速度 rad/s | 最大单刚体净接触力 N | 终止次数 |','|---|---:|---:|---:|---:|']
    for case,rs in groups.items():
        lines.append(f"|{case}: {rs[0]['label']}|{max(float(x['base_speed']) for x in rs):.3f}|{max(float(x['joint_speed']) for x in rs):.3f}|{max(float(x['contact_force']) for x in rs):.1f}|{sum(x['terminated']=='True' for x in rs)}|")
    lines+=['','接触力为子步历史中的单刚体净力模长峰值，不是冲量。短时不终止不能证明初始碰撞几何有效。', '', '[baseline 侧视](baseline_side.png) · [最差样本侧视](worst_side.png) · [1 秒后的最差样本](worst_after_1s.png)']
lines+=['','## 结果解释','',
'0.074 m 消除了当前无扰动 seed 的静态几何穿地，但完整随机化仍有 929/1024 个样本低于 -1 mm。单关节最深重叠来自 Hip Pitch，其次 Knee；Waist Pitch/Yaw 和 Shoulder Pitch 会单独导致躯干/肩部重叠。单独踝关节组合在本次范围与采样内未穿地，但不能推断其在其他腿部角度下没有影响。', '',
'扫描中 Hip Pitch 从 default=-0.312 向较大值移动、Knee 从 default=0.669 向较大值移动会造成脚部重叠；Waist Pitch 向负方向、Shoulder Pitch 向正方向会造成上半身重叠。左右腿全关节组合有 677/1024 个样本低于 -1 mm，全随机为 929/1024，表明只处理腿部仍不足以解决全身问题。', '',
'完整随机姿态中恢复单个 Hip Pitch 的全身最低点平均改善最大（左 6.61 mm，右 5.69 mm），但仍留下 900 多个失败样本。部分恢复操作也会让特定组合变差，不能将单关节扫描范围直接当作已验证的组合安全范围。此次不自动调整训练随机范围。', '',
'所有 146 个条件保存的关节姿态均复核限位，最大越界量为 0 rad，见 joint_limit_check.json；穿地不等于关节限位违规。', '',
'后续应先对齐哪些关节允许不对称扰动、哪些保留双向扰动，再对组合姿态重新做碰撞筛查。当前 1.43 mm 的 nominal 几何间隙也不构成最终物理接触标定。', '',
'## 数据与复现','', '[敏感性图](sensitivity.png) · [扫描曲线](scan_curves.png) · [全部测试指标](metrics.csv) · [恢复消融](ablation.csv)','',
'inputs.npz 保存共享完整随机姿态、nominal、root、主动关节索引和兼容限位。每个测试条件对应一个 npz，保存 q 和逐碰撞体最低高度；形状顺序与关节名称见 results.json。dynamics_initial.npz 保存选定动力学姿态和标签，dynamics.csv 保存逐控制步记录。', '',
'```bash','source /data/g1_hoi_ws/sim51/bin/activate','python scripts/getup/sensitivity_reset.py --headless --root-height .074 --output outputs/getup_reset_sensitivity_074_new','python scripts/getup/report_sensitivity.py --output outputs/getup_reset_sensitivity_074_new','```', '', '脚本拒绝覆盖已有输出目录；USD 路径可用 --usd 指定。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['scripts/getup/sensitivity_reset.py','scripts/getup/imported_geometry.py','source/g1_hoi_learning/g1_hoi_learning/tasks/getup/pose_cfg.py','source/g1_hoi_learning/g1_hoi_learning/tasks/getup/mdp/reset.py']
(out/'provenance.json').write_text(json.dumps({'sha256':{x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in paths}},indent=2))
print('TOP SINGLE',[(r['label'],r['min_z'],r['below_minus_1mm']) for r in singles[:8]])
print('TOP RESTORE',rest[:6]);print('report saved',out/'REPORT.md')
