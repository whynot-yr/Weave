"""Local robust refinement of a laterally separated arm pose."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from imported_geometry import ImportedGeometry
src=Path('outputs/getup_prone_final');s=json.loads((src/'results.json').read_text());e=next(x for x in s['candidates'] if x['label']=='global_arms_0');names=s['names'];drive=[names.index(n) for n in s['drive_names']];q0=np.array(e['joint_pos_53']);root=np.array(e['root_state_13']);limits=np.load('outputs/getup_reset_sensitivity_074/inputs.npz')['limits'];mimic=s['mimic'];g=ImportedGeometry(s['usd'],'source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf');parts=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_pitch','wrist_roll'];x0=np.array([q0[names.index('left_'+n+'_joint')] for n in parts]);bounds=[(max(limits[names.index('left_'+n+'_joint'),0],x-.25),min(limits[names.index('left_'+n+'_joint'),1],x+.25)) for n,x in zip(parts,x0)]
rand=np.load(src/'randomness.npz');u=rand['scale'];v=rand['offset'];rng=np.random.default_rng(31415);corners=rng.choice([-1.,1.],(192,41));arms=[i for i,x in enumerate(g.shapes) if any(n in x[0] for n in ('shoulder','elbow','wrist')) or x[0].startswith(('L_','R_'))];hands=[i for i,x in enumerate(g.shapes) if x[0].startswith(('L_','R_')) or 'wrist_yaw' in x[0]]
def derive(q):
 q[:,drive]=np.clip(q[:,drive],limits[drive,0],limits[drive,1])
 for n,(d,m,b) in mimic.items():q[:,names.index(n)]=q[:,names.index(d)]*m+b
 return q
def decode(x):
 q=q0.copy()
 for side,sign in [('left',1),('right',-1)]:
  for n,val in zip(parts,x):q[names.index(side+'_'+n+'_joint')]=val*(sign if n in ('shoulder_roll','shoulder_yaw','wrist_roll') else 1)
 return derive(q[None])[0]
calls=0
def loss(x):
 global calls
 q=decode(x);qs=np.repeat(q[None],1024,0);qs[:,drive]=q[drive]*u+v;qs=derive(qs);cq=np.repeat(q[None],len(corners),0);cq[:,drive]+=corners*(.1*abs(q[drive])+.1);cq=derive(cq);allq=np.concatenate([q[None],qs,cq]);h=g.minimum_heights(np.repeat(root[None],len(allq),0),allq,names);z=h[:,arms].min(-1);pen=np.maximum(.003-z,0)
 cost=500*pen.max()+200*pen.mean()+30*max(h[0,hands].min()-.12,0)+.1*np.sum((x-x0)**2)
 calls+=1
 if calls%100==0:print('calls',calls,'loss',cost,flush=True)
 return cost
res=minimize(loss,x0,method='Powell',bounds=bounds,options={'maxfev':600,'xtol':.0005,'ftol':.001});q=decode(res.x);e.update(label='refined_arms',joint_pos_53=q.tolist(),active_41=q[drive].tolist(),joint_pos=dict(zip(names,q.tolist())));Path('outputs/getup_prone_exploration/refined_arm_candidate.json').write_text(json.dumps(e,indent=2));Path('outputs/getup_prone_exploration/refined_arm_optimization.json').write_text(json.dumps({'x':res.x.tolist(),'start':x0.tolist(),'loss':res.fun,'nfev':res.nfev,'message':str(res.message)},indent=2));print('DONE',res.fun,flush=True)
