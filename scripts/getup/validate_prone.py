"""Render candidates and compare hold-seed vs zero-action PhysX rollouts."""
import argparse,json,csv
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/getup_prone_exploration'));AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(a).app
import numpy as np
import torch
from PIL import Image
from isaaclab.sim import SimulationContext
from isaaclab.sensors import ContactSensorCfg
from g1_hoi_learning.tasks.getup.env import GetupEnv
from g1_hoi_learning.tasks.getup.env_cfg import GetupEnvCfg
from g1_hoi_learning.assets import G1_INSPIRE_URDF_PATH
from imported_geometry import ImportedGeometry

def main():
 torch.set_num_threads(4);out=a.output;s=json.loads((out/'results.json').read_text());entries=s['candidates'];byname={x['label']:x for x in entries};selected=s['selected'];best=byname[selected]
 cfg=GetupEnvCfg();cfg.scene.num_envs=16;cfg.sim.device=a.device;cfg.pose.randomize_joints=False;cfg.scene.audit_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=4)
 env=GetupEnv(cfg);env.reset();r=env.scene['robot'];drive=env.getup_state.drive;geom=ImportedGeometry(s['usd'],G1_INSPIRE_URDF_PATH)
 def write(cases):
  q=torch.tensor(np.stack([x[1] for x in cases]),device=env.device,dtype=r.data.joint_pos.dtype);root=torch.tensor(np.stack([x[2] for x in cases]),device=env.device,dtype=r.data.root_state_w.dtype);root[:,:3]+=env.scene.env_origins
  r.write_root_pose_to_sim(root[:,:7]);r.write_root_velocity_to_sim(torch.zeros_like(root[:,7:]));r.write_joint_state_to_sim(q,torch.zeros_like(q));r.set_joint_position_target(q);return q
 import omni.replicator.core as rep
 prod=rep.create.render_product('/OmniverseKit_Persp',(960,720));annot=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');annot.attach([prod])
 def pic(i,label,view):
  c=env.scene.env_origins[i].cpu().numpy()+np.array([.15,0,.15]);eyes={'side':[0,2,.3],'oblique':[1.4,1.6,1.4],'top':[0,.02,2.4]};env.sim.set_camera_view(c+np.array(eyes[view]),c);env.sim.forward();r.update(1e-8)
  for _ in range(5):env.sim.render()
  Image.fromarray(np.asarray(annot.get_data())[:,:,:3]).save(out/(label+'.png'))
 initial=[(e['label'],np.array(e['joint_pos_53']),np.array(e['root_state_13'])) for e in entries];write((initial*16)[:16])
 for i,e in enumerate(entries):
  for view in ['side','oblique']:pic(i,e['label']+'_'+view,view)
 pic(next(i for i,e in enumerate(entries) if e['label']==selected),selected+'_top','top')
 print('[PRONE PHYSX] pictures saved',flush=True)
 probes=[(label,np.array(byname[label]['joint_pos_53']),np.array(byname[label]['root_state_13'])) for label in ['host_reference','flipped_supine',selected]]
 data=np.load(out/(selected+'.npz'));order=np.argsort(data['minima'].min(-1))[:5]
 for idx in order:probes.append((selected+'_random_'+str(idx),data['q'][idx],data['root']))
 cases=[(mode+'_'+label,q,root) for mode in ['hold','zero'] for label,q,root in probes];env.reset();q=write(cases);action=q[:,drive]-r.data.default_joint_pos[:,drive];action[8:]=0
 pic(3,'selected_worst_initial','side');rows=[];forces=[];trajectories=[]
 for step in range(50):
  _,_,term,trunc,_=env.step(action);f=env.scene['audit_contacts'].data.net_forces_w_history.norm(dim=-1).amax(dim=1);forces.append(f.cpu().numpy())
  roots=r.data.root_state_w.cpu().numpy().copy();roots[:,:3]-=env.scene.env_origins.cpu().numpy();qs=r.data.joint_pos.cpu().numpy().copy();h=geom.minimum_heights(roots,qs,r.joint_names);trajectories.append((roots,qs))
  for i,(label,_,_) in enumerate(cases):rows.append({'case':i,'label':label,'step':step+1,'pelvis_height':float(roots[i,2]),'base_speed':float(r.data.root_lin_vel_w[i].norm()),'joint_speed':float(r.data.joint_vel[i].abs().max()),'contact_force':float(f[i].max()),'head_geometry_min_z':float(h[i,s['head_shape_indices']].min()),'global_geometry_min_z':float(h[i].min()),'terminated':bool(term[i]),'truncated':bool(trunc[i])})
  if step%10==9:print('[PRONE PHYSX] step',step+1,flush=True)
 for i in [0,1,2,3,8,9,10,11]:pic(i,cases[i][0]+'_after_1s','side')
 with (out/'dynamics.csv').open('w') as file:w=csv.DictWriter(file,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'contact_forces.npz',forces=np.stack(forces),body_names=np.array(env.scene['audit_contacts'].body_names),labels=np.array([x[0] for x in cases]))
 np.savez_compressed(out/'dynamics_initial.npz',q=np.stack([x[1] for x in cases]),root=np.stack([x[2] for x in cases]),labels=np.array([x[0] for x in cases]))
 np.savez_compressed(out/'dynamics_trajectory.npz',root=np.stack([x[0] for x in trajectories]),q=np.stack([x[1] for x in trajectories]))
 annot.detach();env.close();print('[PRONE PHYSX] COMPLETE',flush=True)
try:main()
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
