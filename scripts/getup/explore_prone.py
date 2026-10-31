"""Offline prone seed exploration on the 68 imported USD collision shapes.
Run with Isaac Sim pxr library paths and OPENBLAS_NUM_THREADS=1.
"""
import argparse,ast,json,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from imported_geometry import ImportedGeometry
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/getup_prone_exploration'));p.add_argument('--arms-near-ground',action='store_true');p.add_argument('--usd',default='/tmp/IsaacLab/usd_20261030_215206_1884/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
old=json.loads(Path('outputs/getup_supine_exploration/results.json').read_text());sup=json.loads(Path('outputs/getup_supine_exploration/refined_fresh.json').read_text());names=old['names'];drive=[names.index(x) for x in old['drive_names']];mimic=old['mimic'];limits=np.load('outputs/getup_reset_sensitivity_074/inputs.npz')['limits']
urdf='source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf';geom=ImportedGeometry(a.usd,urdf);assert len(geom.shapes)==68
head=[i for i,x in enumerate(geom.shapes) if '/head_link/' in x[1]];front=[i for i,x in enumerate(geom.shapes) if x[0]=='pelvis' or (x[0]=='torso_link' and i not in head)];chest=[i for i in front if geom.shapes[i][0]=='torso_link'];assert head
base=np.array(sup['joint_pos_53']);root=np.zeros(13)
refpath=Path('/data/g1_hoi_ws/HoST/legged_gym/legged_gym/envs/g1/g1_config_ground_prone.py');tree=ast.parse(refpath.read_text());reference=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='default_joint_angles' for t in n.targets))
reference_q=base.copy()
for n,v in reference.items():
 if n in names:reference_q[names.index(n)]=v

def derive(q):
 q[:,drive]=np.clip(q[:,drive],limits[drive,0],limits[drive,1])
 for n,(d,m,b) in mimic.items():q[:,names.index(n)]=q[:,names.index(d)]*m+b
 return q

def heights(rt,q):return geom.minimum_heights(np.repeat(rt[None],len(q),0),q,names)
def place(q,angle=np.pi/2):
 rt=root.copy();rt[3:7]=[np.cos(angle/2),0,np.sin(angle/2),0];h=heights(rt,q[None])[0];rt[2]=.0015-h[front].min();return rt

def samples(q,u,v):
 qs=np.repeat(q[None],len(u),0);qs[:,drive]=q[drive]*u+v;return derive(qs)
# Symmetric limbs; wrist yaw signs and shoulder yaw signs mirrored.
parts=['hip_pitch','knee','ankle_pitch','shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_pitch','wrist_roll']
bounds=[(-.15,.65),(0.,.85),(-.45,.35),(-.8,1.5),(.05,1.1),(-.8,.8),(0.,1.3),(-.45,.45),(-.5,.5),(-.48,.05),(1.42,1.72)]
hands=[i for i,x in enumerate(geom.shapes) if x[0].startswith(('L_','R_')) or 'wrist_yaw' in x[0]]
if a.arms_near_ground:
 bounds[3]=(-.3,.8);bounds[4]=(.2,1.5);bounds[5]=(-2.,2.);bounds[6]=(.1,2.05)

def decode(x):
 q=reference_q.copy()
 for side,sign in [('left',1),('right',-1)]:
  for j,part in enumerate(parts):q[names.index(side+'_'+part+'_joint')]=x[j]*(sign if part in ('shoulder_roll','shoulder_yaw','wrist_roll') else 1)
 q[names.index('waist_pitch_joint')]=x[9];q=derive(q[None])[0];return q,place(q,x[10])
rng=np.random.default_rng(2026110101);ou=rng.uniform(.9,1.1,(96,41));ov=rng.uniform(-.1,.1,(96,41));corners=rng.choice([-1.,1.],(128,41));logs=[];calls=0

def objective(x):
 global calls
 q,rt=decode(x);qs=samples(q,ou,ov);cq=np.repeat(q[None],len(corners),0);cq[:,drive]+=corners*(.1*np.abs(q[drive])+.1);cq=derive(cq)
 h=heights(rt,np.concatenate([q[None],qs,cq]));z=h.min(-1);pen=np.maximum(.0005-z,0)
 cost=200*pen[1:].mean()+150*pen[1:].max()+2000*pen[0]
 # Nominal front support is mandatory; head must not be the supporting surface.
 cost+=1000*max(rt[2]-.12,0)+1000*max(.04-rt[2],0)+400*max(h[0,chest].min()-.008,0)
 cost+=200*max(.015-h[0,head].min(),0)+100*max(.005-h[:,head].min(),0)
 if a.arms_near_ground:cost+=100*max(h[0,hands].min()-.05,0)
 cost+=.003*np.sum((x[:9]-np.array([.15,.25,0,.3,.5,0,.4,0,0]))**2)
 calls+=1
 if calls%150==0:print('[PRONE] calls',calls,'loss',float(cost),flush=True)
 return float(cost)
starts=[[.15,.25,0,.3,.5,0,.4,0,0,-.25,np.pi/2],[.25,.4,.1,.8,.7,.2,.6,0,0,-.35,1.6],[.1,.1,0,-.3,.9,-.3,.2,0,0,-.2,1.5]]
if a.arms_near_ground:
 starts=[[.44,.25,0,.1,.5,1.57,.8,0,0,-.344,1.5995],[.44,.25,0,.1,.5,-1.57,.8,0,0,-.344,1.5995],[.44,.25,0,.4,1.,1.57,1.4,0,0,-.344,1.5995]]
candidates=[('host_reference',reference_q,place(reference_q)),('flipped_supine',base,place(base))]
for i,x in enumerate(starts):
 q,rt=decode(x);candidates.append(('constructed_'+str(i),q,rt))
 result=minimize(objective,x,method='Powell',bounds=bounds,options={'maxfev':1000,'xtol':.003,'ftol':.002});q,rt=decode(result.x);candidates.append(('optimized_'+str(i),q,rt));logs.append({'label':'optimized_'+str(i),'x':result.x.tolist(),'loss':float(result.fun),'nfev':result.nfev,'message':str(result.message)})
 print('[PRONE] optimized',i,result.fun,'root',rt[2],flush=True)
(out/'optimization.json').write_text(json.dumps(logs,indent=2));rng=np.random.default_rng(2026110102);u=rng.uniform(.9,1.1,(1024,41));v=rng.uniform(-.1,.1,(1024,41));np.savez_compressed(out/'randomness.npz',scale=u,offset=v,fit_scale=ou,fit_offset=ov,fit_corner_signs=corners)
result={'names':names,'drive_names':old['drive_names'],'mimic':mimic,'usd':a.usd,'head_shape_indices':head,'front_shape_indices':front,'chest_shape_indices':chest,'shapes':[{'body':x[0],'path':x[1]} for x in geom.shapes],'validation_seed':2026110102,'candidates':[]}
for label,q,rt in candidates:
 qs=samples(q,u,v);h=heights(rt,qs);nh=heights(rt,q[None])[0];points=geom.lowest_points(rt[None],q[None],names)[0];assert np.max(abs(points[:,2]-nh))<1e-8
 allpoints=geom.lowest_points(np.repeat(rt[None],1024,0),qs,names);minimum_points=allpoints[np.arange(1024),h.argmin(-1)]
 e={'label':label,'root_pos':rt[:3].tolist(),'root_quat_wxyz':rt[3:7].tolist(),'root_state_13':rt.tolist(),'joint_pos':dict(zip(names,q.tolist())),'active_41':q[drive].tolist(),'joint_pos_53':q.tolist(),'joint_vel_53':[0.]*53,
 'nominal_min_z':float(nh.min()),'nominal_chest_min_z':float(nh[chest].min()),'nominal_head_min_z':float(nh[head].min()),'nominal_lowest_points':points.tolist(),
 'valid_no_penetration':int((h.min(-1)>=0).sum()),'valid_1mm_tolerance':int((h.min(-1)>=-.001).sum()),'random_min_z':float(h.min()),'median_min_z':float(np.median(h.min(-1))),
 'random_head_min_z':float(h[:,head].min()),'near_ground_count_per_shape':((h>=-.001)&(h<=.005)).sum(0).tolist(),'random_front_near_ground':int(((h[:,front]>=-.001)&(h[:,front]<=.005)).any(-1).sum()),
 'joint_limit_violation':float(np.maximum(limits[:,0]-qs,qs-limits[:,1]).clip(0).max()),'mimic_error':max(float(abs(qs[:,names.index(n)]-qs[:,names.index(d)]*m-b).max()) for n,(d,m,b) in mimic.items())}
 e['nominal_constraints_passed']=bool(.04<=rt[2]<=.12 and nh.min()>=0 and nh[chest].min()<=.008 and nh[head].min()>=.015)
 result['candidates'].append(e);(out/(label+'.json')).write_text(json.dumps(e,indent=2));np.savez_compressed(out/(label+'.npz'),q=qs,root=rt,minima=h,lowest_points=minimum_points,nominal_minima=nh)
 print('[PRONE] validation',label,e['valid_no_penetration'],e['random_min_z'],'head',e['random_head_min_z'],'constraints',e['nominal_constraints_passed'],flush=True)
eligible=[e for e in result['candidates'] if e['nominal_constraints_passed']];best=max(eligible or result['candidates'],key=lambda x:(x['valid_no_penetration'],x['random_min_z']));result['selected']=best['label'];result['selection_meets_nominal_constraints']=bool(eligible)
(out/'results.json').write_text(json.dumps(result,indent=2));print('[PRONE] COMPLETE selected',best['label'],flush=True)
