"""Explore supine seed poses on the imported WEAVE collision geometry."""
import argparse,ast,json,csv,time
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('outputs/getup_supine_exploration'))
p.add_argument('--usd',default='/tmp/IsaacLab/usd_20261030_215206_1884/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=True
app=AppLauncher(args).app
import numpy as np
import torch
from scipy.optimize import minimize
from PIL import Image
from isaaclab.sim import SimulationContext
from isaaclab.sensors import ContactSensorCfg
from g1_hoi_learning.tasks.getup.env import GetupEnv
from g1_hoi_learning.tasks.getup.env_cfg import GetupEnvCfg
from g1_hoi_learning.assets import G1_INSPIRE_URDF_PATH
from imported_geometry import ImportedGeometry

def main():
 out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 cfg=GetupEnvCfg();cfg.scene.num_envs=16;cfg.sim.device=args.device;cfg.pose.randomize_joints=False
 cfg.scene.audit_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=4)
 env=GetupEnv(cfg);env.reset();s=env.getup_state;r=s.robot;names=r.joint_names;drive=list(s.drive);mimic=s.action.cfg.mimic
 # Preserve the historical standing-angles baseline after promoting a new training seed.
 base=r.data.default_joint_pos[0].cpu().numpy().astype(float)
 root=np.array([0.,0.,.06,np.sqrt(.5),0.,-np.sqrt(.5),0.,0.,0.,0.,0.,0.,0.])
 geom=ImportedGeometry(args.usd,G1_INSPIRE_URDF_PATH);assert len(geom.shapes)==68
 limits=r.data.joint_pos_limits[0].cpu().numpy().astype(float)
 for name,(driver,m,b) in mimic.items():
  j,d=names.index(name),names.index(driver);v=(limits[j]-b)/m;limits[d,0]=max(limits[d,0],v.min());limits[d,1]=min(limits[d,1],v.max())
 def derive(q):
  q[:,drive]=np.clip(q[:,drive],limits[drive,0],limits[drive,1])
  for name,(driver,m,b) in mimic.items():q[:,names.index(name)]=q[:,names.index(driver)]*m+b
  return q
 tree=ast.parse(Path('/data/g1_hoi_ws/HoST/legged_gym/legged_gym/envs/g1/g1_config_ground.py').read_text())
 ref=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='default_joint_angles' for t in n.targets))
 host=base.copy()
 for name,value in ref.items():
  if name in names:host[names.index(name)]=value
 host=derive(host[None])[0]
 pelvis=[i for i,x in enumerate(geom.shapes) if x[0]=='pelvis'];torso=[i for i,x in enumerate(geom.shapes) if x[0]=='torso_link'];back=pelvis+torso
 rng=np.random.default_rng(20261031);u=rng.uniform(.9,1.1,(1024,41));v=rng.uniform(-.1,.1,(1024,41))
 opt_rng=np.random.default_rng(9876);ou=opt_rng.uniform(.9,1.1,(64,41));ov=opt_rng.uniform(-.1,.1,(64,41))
 def randomize(q,u,v):
  qs=np.repeat(q[None],len(u),0);qs[:,drive]=q[drive]*u+v;return derive(qs)
 def heights(rt,q):return geom.minimum_heights(np.repeat(rt[None],len(q),0),q,names)
 def attach_back(q,angle=-np.pi/2):
  rt=root.copy();rt[:3]=0;rt[3:7]=[np.cos(angle/2),0,np.sin(angle/2),0];h=heights(rt,q[None])[0]
  rt[2]=.0015-h[back].min();return rt
 # Anatomically modest, symmetric search; root height is derived from back support, not optimized independently.
 bounds=[(-.7,-.08),(.05,1.1),(-.5,.15),(-.6,.1),(.05,.65),(.35,1.25),(-.03,.35),(-1.72,-1.42)]
 def decode(x):
  q=host.copy()
  for side,sign in [('left',1),('right',-1)]:
   for part,val in zip(['hip_pitch','knee','ankle_pitch','shoulder_pitch','shoulder_roll','elbow'],[x[0],x[1],x[2],x[3],sign*x[4],x[5]]):q[names.index(side+'_'+part+'_joint')]=val
  q[names.index('waist_pitch_joint')]=x[6];q=derive(q[None])[0];return q,attach_back(q,x[7])
 calls=[0]
 def objective(x):
  q,rt=decode(x);h=heights(rt,np.concatenate([q[None],randomize(q,ou,ov)]));z=h.min(-1);pen=np.maximum(.0005-z,0)
  # Keep the trunk close to the floor, penalize nominal overlap, and prevent high-root solutions.
  cost=pen[1:].mean()*200+pen[1:].max()*80+pen[0]*2000
  cost+=max(rt[2]-.10,0)*1000+max(.045-rt[2],0)*1000
  cost+=max(h[0,torso].min()-.008,0)*400
  cost+=.01*np.sum((x[:6]-np.array([-.3,.45,-.2,-.1,.25,.8]))**2)
  calls[0]+=1
  if calls[0]%150==0:print('[EXP] optimize calls',calls[0],'loss',float(cost),flush=True)
  return float(cost)
 candidates=[('baseline_060',base,root.copy()),('baseline_074',base,root.copy()),('host_reference',host,attach_back(host))]
 candidates[1][2][2]=.074
 starts=[[-.3,.45,-.2,-.1,.25,.8,.1,-np.pi/2],[-.5,.7,-.2,-.3,.4,.8,.2,-1.6],[-.2,.15,-.1,-.2,.2,.8,.15,-1.5]]
 logs=[]
 for i,x in enumerate(starts):
  result=minimize(objective,x,method='Powell',bounds=bounds,options={'maxfev':500,'xtol':.004,'ftol':.002})
  q,rt=decode(result.x);candidates.append(('optimized_'+str(i),q,rt));logs.append({'x':result.x.tolist(),'fun':float(result.fun),'nfev':result.nfev,'message':str(result.message)})
  print('[EXP] candidate',i,result.fun,'root',rt[2],flush=True)
 (out/'optimization.json').write_text(json.dumps(logs,indent=2))
 metadata={'names':names,'drive_names':[names[i] for i in drive],'mimic':mimic,'usd':args.usd,'shapes':[{'body':x[0],'path':x[1]} for x in geom.shapes],'reference_angles':ref,'optimization_samples':64,'validation_seed':20261031,'candidates':[]}
 np.savez_compressed(out/'shared_randomness.npz',scale=u,offset=v,optimization_scale=ou,optimization_offset=ov)
 for label,q,rt in candidates:
  qs=randomize(q,u,v);h=heights(rt,qs);nom=heights(rt,q[None])[0];z=h.min(-1)
  valid=z>=0;valid_tol=z>=-.001
  near=(h>=-.001)&(h<=.005)
  entry={'label':label,'root_pos':rt[:3].tolist(),'root_quat_wxyz':rt[3:7].tolist(),'joint_pos':dict(zip(names,q.tolist())),
    'active_41':q[drive].tolist(),'joint_pos_53':q.tolist(),'joint_vel_53':[0.]*53,'root_state_13':rt.tolist(),
    'nominal_min_z':float(nom.min()),'nominal_back_min_z':float(nom[back].min()),'nominal_torso_min_z':float(nom[torso].min()),
    'nominal_near_ground_shapes':[geom.shapes[i][1] for i in np.where((nom>=-.001)&(nom<=.005))[0]],
    'valid_no_penetration':int(valid.sum()),'valid_1mm_tolerance':int(valid_tol.sum()),'random_min_z':float(z.min()),'median_min_z':float(np.median(z)),
    'random_back_near_ground':int(near[:,back].any(-1).sum()),'random_torso_near_ground':int(near[:,torso].any(-1).sum()),
    'near_ground_count_per_shape':near.sum(0).tolist(),'joint_limit_violation':float(np.maximum(limits[:,0]-qs,qs-limits[:,1]).clip(0).max()),
    'mimic_error':max(float(np.abs(qs[:,names.index(n)]-qs[:,names.index(d)]*m-b).max()) for n,(d,m,b) in mimic.items())}
  metadata['candidates'].append(entry);(out/(label+'.json')).write_text(json.dumps(entry,indent=2));np.savez_compressed(out/(label+'.npz'),q=qs,root=rt,minima=h,nominal_minima=nom)
  print('[EXP] validation',label,entry['valid_no_penetration'],entry['valid_1mm_tolerance'],entry['random_min_z'],flush=True)
 (out/'results.json').write_text(json.dumps(metadata,indent=2))
 # Validate every nominal candidate and baseline/best randomized cases in one simulation.
 eligible=[e for e in metadata['candidates'][2:] if .045<=e['root_pos'][2]<=.10 and e['nominal_min_z']>=-.001 and e['nominal_torso_min_z']<=.008]
 best=max(eligible,key=lambda e:(e['valid_no_penetration'],e['random_min_z'])) if eligible else max(metadata['candidates'][2:],key=lambda e:e['valid_no_penetration'])
 metadata['selected']=best['label'];metadata['selection_meets_nominal_constraints']=bool(eligible)
 cases=[(label,q,rt) for label,q,rt in candidates]
 for label in ['baseline_060',best['label']]:
  qfile=np.load(out/(label+'.npz'));zs=qfile['minima'].min(-1)
  for rank,idx in enumerate([int(zs.argmin()),int(np.argsort(zs)[512]),0,1,2]):cases.append((label+'_random_'+str(idx),qfile['q'][idx],qfile['root']))
 assert len(cases)==16
 env.reset();qt=torch.tensor(np.stack([x[1] for x in cases]),device=env.device,dtype=r.data.joint_pos.dtype)
 roots=torch.tensor(np.stack([x[2] for x in cases]),device=env.device,dtype=r.data.root_state_w.dtype);roots[:,:3]+=env.scene.env_origins
 r.write_root_pose_to_sim(roots[:,:7]);r.write_root_velocity_to_sim(torch.zeros_like(roots[:,7:]));r.write_joint_state_to_sim(qt,torch.zeros_like(qt));r.set_joint_position_target(qt)
 action=qt[:,drive]-r.data.default_joint_pos[:,drive]
 import omni.replicator.core as rep
 prod=rep.create.render_product('/OmniverseKit_Persp',(960,720));annot=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');annot.attach([prod])
 def picture(i,tag,view='side'):
  center=env.scene.env_origins[i].cpu().numpy()+np.array([-.15,0,.15]);eye=np.array([0,2,.3]) if view=='side' else np.array([1.4,1.6,1.4])
  env.sim.set_camera_view(center+eye,center);env.sim.forward();r.update(1e-8)
  for _ in range(5):env.sim.render()
  Image.fromarray(np.asarray(annot.get_data())[:,:,:3]).save(out/(tag+'.png'))
 for i,(label,_,_) in enumerate(cases[:6]):
  picture(i,label+'_side');picture(i,label+'_oblique','oblique')
 picture(11,'best_worst_random_initial')
 rows=[];contacts=[]
 for step in range(50):
  _,_,term,trunc,_=env.step(action);f=env.scene['audit_contacts'].data.net_forces_w_history.norm(dim=-1).amax(dim=1)
  contacts.append(f.cpu().numpy())
  for i,(label,_,_) in enumerate(cases):rows.append({'case':i,'label':label,'step':step+1,'pelvis_height':float(r.data.root_pos_w[i,2]-env.scene.env_origins[i,2]),'base_speed':float(r.data.root_lin_vel_w[i].norm()),'joint_speed':float(r.data.joint_vel[i].abs().max()),'contact_force':float(f[i].max()),'terminated':bool(term[i]),'truncated':bool(trunc[i])})
 for i in [0,1,2,3,4,5,11]:picture(i,cases[i][0]+'_after_1s')
 with (out/'dynamics.csv').open('w') as file:w=csv.DictWriter(file,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'contact_forces.npz',forces=np.stack(contacts),body_names=np.array(env.scene['audit_contacts'].body_names),labels=np.array([x[0] for x in cases]))
 metadata['dynamics_cases']=[x[0] for x in cases];(out/'results.json').write_text(json.dumps(metadata,indent=2));annot.detach();env.close();print('[EXP] COMPLETE',out,flush=True)
try:main()
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
