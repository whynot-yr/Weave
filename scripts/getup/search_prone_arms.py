"""Global arm search to avoid poor local minima, with lateral wrist separation."""
import json
from pathlib import Path
import numpy as np
from imported_geometry import ImportedGeometry
out=Path('outputs/getup_prone_exploration');s=json.loads((out/'results.json').read_text());best=next(e for e in s['candidates'] if e['label']==s['selected']);names=s['names'];drive=[names.index(n) for n in s['drive_names']];mimic=s['mimic'];q0=np.array(best['joint_pos_53']);root=np.array(best['root_state_13']);limits=np.load('outputs/getup_reset_sensitivity_074/inputs.npz')['limits'];g=ImportedGeometry(s['usd'],'source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf')
hand=[i for i,x in enumerate(g.shapes) if x[0].startswith(('L_','R_')) or 'wrist_yaw' in x[0]]
def derive(q):
 q[:,drive]=np.clip(q[:,drive],limits[drive,0],limits[drive,1])
 for n,(d,m,b) in mimic.items():q[:,names.index(n)]=q[:,names.index(d)]*m+b
 return q
rng=np.random.default_rng(2026110103);qs=np.repeat(q0[None],12000,0)
parts=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_pitch','wrist_roll'];lo=np.array([-.6,.2,-2.,.1,-.7,-1.]);hi=np.array([.8,1.4,2.,2.05,.7,1.]);params=rng.uniform(lo,hi,(len(qs),6))
for side,sign in [('left',1),('right',-1)]:
 for j,part in enumerate(parts):qs[:,names.index(side+'_'+part+'_joint')]=params[:,j]*(sign if part in ('shoulder_roll','shoulder_yaw','wrist_roll') else 1)
qs=derive(qs);roots=np.repeat(root[None],len(qs),0);h=g.minimum_heights(roots,qs,names);fk=g.source.transforms(roots[:,:3],roots[:,3:7],qs,names)
mask=(h.min(-1)>=.0005)&(h[:,hand].min(-1)<.12)&(fk['left_wrist_yaw_link'][:,1,3]>.16)&(fk['right_wrist_yaw_link'][:,1,3]<-.16)
ids=np.where(mask)[0];print('nominal screened',len(ids),flush=True)
# Test shortlisted poses with shared fit perturbations; neither these nor this table are final validation.
fit_u=rng.uniform(.9,1.1,(256,41));fit_v=rng.uniform(-.1,.1,(256,41));scores=[]
for j,idx in enumerate(ids[:500]):
 q=np.repeat(qs[idx,None],256,0);q[:,drive]=qs[idx,drive]*fit_u+fit_v;q=derive(q);hs=g.minimum_heights(np.repeat(root[None],256,0),q,names);z=hs.min(-1)
 scores.append((int((z>=0).sum()),float(z.min()),float(h[idx,hand].min()),int(idx)))
 if j%50==0:print('fit',j,'best',max(scores),flush=True)
# Prefer all-fit-passing low-arm candidates, then worst-case geometry among ties.
scores.sort(key=lambda x:(x[0],-x[2] if x[0]==256 else x[1]),reverse=True)
(out/'arm_global_search.json').write_text(json.dumps({'sampled':12000,'nominal_pass_count':len(ids),'tested':len(scores),'top':scores[:10],'params':params.tolist()},indent=2))
if not scores:raise RuntimeError('No separated-arm candidates')
# Keep three alternatives for a new independent 1024 comparison.
candidates=[]
for rank,score in enumerate(scores[:3]):
 q=qs[score[-1]];e=dict(best);e.update(label='global_arms_'+str(rank),joint_pos_53=q.tolist(),active_41=q[drive].tolist(),joint_pos=dict(zip(names,q.tolist())));candidates.append(e)
(out/'global_arm_candidates.json').write_text(json.dumps(candidates,indent=2));print('DONE',scores[:3],flush=True)
