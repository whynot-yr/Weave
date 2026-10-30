"""Refine against prior failures and joint-range corners, validate with a fresh seed."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from imported_geometry import ImportedGeometry
out=Path('outputs/getup_supine_exploration');s=json.loads((out/'results.json').read_text());names=s['names'];drive=[names.index(x) for x in s['drive_names']];mimic=s['mimic']
geom=ImportedGeometry(s['usd'],'source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf')
limits=np.load('outputs/getup_reset_sensitivity_074/inputs.npz')['limits'];seed=s['candidates'][3];nom=np.array(seed['joint_pos_53']);root=np.array(seed['root_state_13'])
back=[i for i,x in enumerate(geom.shapes) if x[0] in ('pelvis','torso_link')];torso=[i for i,x in enumerate(geom.shapes) if x[0]=='torso_link']
rand=np.load(out/'shared_randomness.npz');rng=np.random.default_rng(87654)
signs=rng.choice([-1.,1.],(128,41))
# Prior failures become optimization data; fresh validation below remains separate.
u=rand['scale'];v=rand['offset'];calls=0

def derive(q):
 q[:,drive]=np.clip(q[:,drive],limits[drive,0],limits[drive,1])
 for name,(d,m,b) in mimic.items():q[:,names.index(name)]=q[:,names.index(d)]*m+b
 return q

def decode(x):
 q=nom.copy()
 for side,sign in [('left',1),('right',-1)]:
  for part,val in zip(['hip_pitch','knee','ankle_pitch','shoulder_pitch','shoulder_roll','elbow'],[x[0],x[1],x[2],x[3],sign*x[4],x[5]]):q[names.index(side+'_'+part+'_joint')]=val
 q[names.index('waist_pitch_joint')]=x[6];q=derive(q[None])[0];rt=root.copy();rt[2]=0;rt[3:7]=[np.cos(x[7]/2),0,np.sin(x[7]/2),0]
 h=geom.minimum_heights(rt[None],q[None],names)[0];rt[2]=.0015-h[back].min();return q,rt

def samples(q,u,v):
 qs=np.repeat(q[None],len(u),0);qs[:,drive]=q[drive]*u+v;return derive(qs)

def loss(x):
 global calls
 q,rt=decode(x);qs=samples(q,u,v);corners=np.repeat(q[None],len(signs),0);corners[:,drive]+=signs*(.1*np.abs(q[drive])+.1);corners=derive(corners)
 allq=np.concatenate([q[None],qs,corners]);h=geom.minimum_heights(np.repeat(rt[None],len(allq),0),allq,names);z=h.min(-1);p=np.maximum(.0005-z,0)
 cost=200*p[1:].mean()+200*p[1:].max()+2000*p[0]+1000*max(rt[2]-.10,0)+1000*max(.045-rt[2],0)+400*max(h[0,torso].min()-.008,0)
 cost+=.005*np.sum((x[:6]-np.array([-.45,.45,-.2,-.1,.25,.8]))**2)
 calls+=1
 if calls%100==0:print('calls',calls,'cost',cost,flush=True)
 return cost
x=np.array(json.loads((out/'optimization.json').read_text())[0]['x']);x[0]=-.48;x[6]=.34
res=minimize(loss,x,method='Powell',bounds=[(-.7,-.08),(.05,1.1),(-.5,.15),(-.6,.1),(.05,.65),(.35,1.25),(.0,.45),(-1.72,-1.42)],options={'maxfev':650,'xtol':.003,'ftol':.002})
q,rt=decode(res.x);(out/'refinement_optimization.json').write_text(json.dumps({'x':res.x.tolist(),'fun':float(res.fun),'nfev':res.nfev,'message':str(res.message),'fit_samples':1024,'range_corner_samples':128},indent=2))
# Evaluate ALL candidates with the same new table to avoid selection-set bias.
rng=np.random.default_rng(20261101);vu=rng.uniform(.9,1.1,(1024,41));vv=rng.uniform(-.1,.1,(1024,41));np.savez_compressed(out/'fresh_randomness.npz',scale=vu,offset=vv)
new={'label':'refined','root_state_13':rt.tolist(),'root_pos':rt[:3].tolist(),'root_quat_wxyz':rt[3:7].tolist(),'joint_pos_53':q.tolist(),'active_41':q[drive].tolist(),'joint_pos':dict(zip(names,q.tolist())),'joint_vel_53':[0.]*53}
entries=[]
for e in s['candidates']+[new]:
 label=e['label'];cq=np.array(e['joint_pos_53']);cr=np.array(e['root_state_13']);qs=samples(cq,vu,vv);h=geom.minimum_heights(np.repeat(cr[None],1024,0),qs,names);nh=geom.minimum_heights(cr[None],cq[None],names)[0];points=geom.lowest_points(cr[None],cq[None],names)[0]
 assert np.max(np.abs(points[:,2]-nh))<1e-8
 ix=h.argmin(-1);allpoints=geom.lowest_points(np.repeat(cr[None],1024,0),qs,names);mins=allpoints[np.arange(1024),ix]
 e=dict(e);e.update({'nominal_min_z':float(nh.min()),'nominal_torso_min_z':float(nh[torso].min()),'valid_no_penetration':int((h.min(-1)>=0).sum()),'valid_1mm_tolerance':int((h.min(-1)>=-.001).sum()),'random_min_z':float(h.min()),'median_min_z':float(np.median(h.min(-1))),
 'joint_limit_violation':float(np.maximum(limits[:,0]-qs,qs-limits[:,1]).clip(0).max()),'mimic_error':max(float(np.abs(qs[:,names.index(n)]-qs[:,names.index(d)]*m-b).max()) for n,(d,m,b) in mimic.items()),
 'nominal_lowest_points':points.tolist(),'random_back_near_ground':int(((h[:,back]>=-.001)&(h[:,back]<=.005)).any(-1).sum()),'random_torso_near_ground':int(((h[:,torso]>=-.001)&(h[:,torso]<=.005)).any(-1).sum()),'near_ground_count_per_shape':((h>=-.001)&(h<=.005)).sum(0).tolist()})
 entries.append(e);np.savez_compressed(out/(label+'_fresh.npz'),q=qs,root=cr,minima=h,lowest_points=mins,nominal_minima=nh);(out/(label+'_fresh.json')).write_text(json.dumps(e,indent=2))
 print(label,e['valid_no_penetration'],e['valid_1mm_tolerance'],e['random_min_z'],flush=True)
(out/'fresh_results.json').write_text(json.dumps({'names':names,'drive_names':s['drive_names'],'shapes':s['shapes'],'validation_seed':20261101,'candidates':entries},indent=2))
