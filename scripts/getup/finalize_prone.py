"""Independent comparison after global arm screening, preserving all prior data."""
from pathlib import Path
import argparse,json,numpy as np
from imported_geometry import ImportedGeometry
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/getup_prone_final'));p.add_argument('--extra-json',type=Path);p.add_argument('--seed',type=int,default=2026110104);a=p.parse_args()
source=Path('outputs/getup_prone_exploration');out=a.output;out.mkdir(exist_ok=False)
s=json.loads((source/'results.json').read_text());names=s['names'];drive=[names.index(x) for x in s['drive_names']];mimic=s['mimic'];limits=np.load('outputs/getup_reset_sensitivity_074/inputs.npz')['limits'];g=ImportedGeometry(s['usd'],'source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf')
candidates=[e for e in s['candidates'] if e['label'] in ('host_reference','flipped_supine','optimized_1')]+json.loads((source/'global_arm_candidates.json').read_text())
if a.extra_json:candidates.append(json.loads(a.extra_json.read_text()))
rng=np.random.default_rng(a.seed);u=rng.uniform(.9,1.1,(1024,41));v=rng.uniform(-.1,.1,(1024,41));np.savez_compressed(out/'randomness.npz',scale=u,offset=v)
entries=[];hands=[i for i,x in enumerate(g.shapes) if x[0].startswith(('L_','R_')) or 'wrist_yaw' in x[0]]
for e in candidates:
 e=dict(e);q=np.array(e['joint_pos_53']);root=np.array(e['root_state_13']);qs=np.repeat(q[None],1024,0);qs[:,drive]=np.clip(q[drive]*u+v,limits[drive,0],limits[drive,1])
 for n,(d,m,b) in mimic.items():qs[:,names.index(n)]=qs[:,names.index(d)]*m+b
 roots=np.repeat(root[None],1024,0);h=g.minimum_heights(roots,qs,names);nh=g.minimum_heights(root[None],q[None],names)[0];pts=g.lowest_points(root[None],q[None],names)[0];allpts=g.lowest_points(roots,qs,names);minimum=allpts[np.arange(1024),h.argmin(-1)]
 e.update(nominal_min_z=float(nh.min()),nominal_chest_min_z=float(nh[s['chest_shape_indices']].min()),nominal_head_min_z=float(nh[s['head_shape_indices']].min()),nominal_hand_min_z=float(nh[hands].min()),nominal_lowest_points=pts.tolist(),valid_no_penetration=int((h.min(-1)>=0).sum()),valid_1mm_tolerance=int((h.min(-1)>=-.001).sum()),random_min_z=float(h.min()),median_min_z=float(np.median(h.min(-1))),random_head_min_z=float(h[:,s['head_shape_indices']].min()),near_ground_count_per_shape=((h>=-.001)&(h<=.005)).sum(0).tolist(),random_front_near_ground=int(((h[:,s['front_shape_indices']]>=-.001)&(h[:,s['front_shape_indices']]<=.005)).any(-1).sum()),joint_limit_violation=float(np.maximum(limits[:,0]-qs,qs-limits[:,1]).clip(0).max()),mimic_error=max(float(abs(qs[:,names.index(n)]-qs[:,names.index(d)]*m-b).max()) for n,(d,m,b) in mimic.items()))
 e['nominal_constraints_passed']=bool(.04<=root[2]<=.12 and nh.min()>=0 and e['nominal_chest_min_z']<=.008 and e['nominal_head_min_z']>=.015)
 fk=g.source.transforms(root[None,:3],root[None,3:7],q[None],names)
 e['left_wrist_position']=fk['left_wrist_yaw_link'][0,:3,3].tolist();e['right_wrist_position']=fk['right_wrist_yaw_link'][0,:3,3].tolist()
 np.savez_compressed(out/(e['label']+'.npz'),q=qs,root=root,minima=h,lowest_points=minimum,nominal_minima=nh);(out/(e['label']+'.json')).write_text(json.dumps(e,indent=2));entries.append(e);print(e['label'],e['valid_no_penetration'],e['random_min_z'],'hand',e['nominal_hand_min_z'],flush=True)
s['candidates']=entries;s['validation_seed']=a.seed;s['selected']=max(entries,key=lambda e:(e['valid_no_penetration'],-e['nominal_hand_min_z']))['label'];s['selection_meets_nominal_constraints']=next(e['nominal_constraints_passed'] for e in entries if e['label']==s['selected']);(out/'results.json').write_text(json.dumps(s,indent=2));print('SELECTED',s['selected'])
