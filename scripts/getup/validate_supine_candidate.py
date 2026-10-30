"""PhysX comparison of refined supine and unchanged baselines."""
import argparse,json,csv
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/getup_supine_exploration'));AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;app=AppLauncher(a).app
import numpy as np
import torch
from PIL import Image
from isaaclab.sim import SimulationContext
from isaaclab.sensors import ContactSensorCfg
from g1_hoi_learning.tasks.getup.env import GetupEnv
from g1_hoi_learning.tasks.getup.env_cfg import GetupEnvCfg

def main():
 torch.set_num_threads(4);out=a.output;data=json.loads((out/'fresh_results.json').read_text());byname={x['label']:x for x in data['candidates']};best=byname['refined']
 cases=[]
 for name in ['baseline_060','baseline_074','refined']:
  e=byname[name];cases.append((name,np.array(e['joint_pos_53']),np.array(e['root_state_13'])))
 d=np.load(out/'refined_fresh.npz');order=np.argsort(d['minima'].min(-1))
 for i in order[:5]:cases.append(('refined_random_'+str(i),d['q'][i],d['root']))
 cfg=GetupEnvCfg();cfg.scene.num_envs=8;cfg.sim.device=a.device;cfg.pose.randomize_joints=False
 cfg.scene.audit_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=4)
 env=GetupEnv(cfg);env.reset();r=env.scene['robot'];drive=env.getup_state.drive
 qt=torch.tensor(np.stack([x[1] for x in cases]),device=env.device,dtype=r.data.joint_pos.dtype);roots=torch.tensor(np.stack([x[2] for x in cases]),device=env.device,dtype=r.data.root_state_w.dtype);roots[:,:3]+=env.scene.env_origins
 r.write_root_pose_to_sim(roots[:,:7]);r.write_root_velocity_to_sim(torch.zeros_like(roots[:,7:]));r.write_joint_state_to_sim(qt,torch.zeros_like(qt));r.set_joint_position_target(qt)
 action=qt[:,drive]-r.data.default_joint_pos[:,drive]
 import omni.replicator.core as rep
 prod=rep.create.render_product('/OmniverseKit_Persp',(960,720));annot=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');annot.attach([prod])
 def pic(i,label,view):
  c=env.scene.env_origins[i].cpu().numpy()+np.array([-.15,0,.15]);eyes={'side':[0,2,.3],'oblique':[1.4,1.6,1.4],'top':[0,.02,2.4]};env.sim.set_camera_view(c+np.array(eyes[view]),c);env.sim.forward();r.update(1e-8)
  for _ in range(5):env.sim.render()
  Image.fromarray(np.asarray(annot.get_data())[:,:,:3]).save(out/(label+'.png'))
 for view in ['side','oblique','top']:pic(2,'refined_'+view,view)
 pic(3,'refined_worst_initial','side');rows=[];forces=[]
 for step in range(50):
  _,_,term,trunc,_=env.step(action);f=env.scene['audit_contacts'].data.net_forces_w_history.norm(dim=-1).amax(dim=1);forces.append(f.cpu().numpy())
  for i,(label,_,_) in enumerate(cases):rows.append({'case':i,'label':label,'step':step+1,'pelvis_height':float(r.data.root_pos_w[i,2]-env.scene.env_origins[i,2]),'base_speed':float(r.data.root_lin_vel_w[i].norm()),'joint_speed':float(r.data.joint_vel[i].abs().max()),'contact_force':float(f[i].max()),'terminated':bool(term[i]),'truncated':bool(trunc[i])})
  if step%10==9:print('[VALIDATE] step',step+1,flush=True)
 for i in [0,1,2,3]:pic(i,cases[i][0]+'_refined_run_after_1s','side')
 with (out/'refined_dynamics.csv').open('w') as file:w=csv.DictWriter(file,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'refined_contacts.npz',forces=np.stack(forces),body_names=np.array(env.scene['audit_contacts'].body_names),labels=np.array([x[0] for x in cases]))
 np.savez_compressed(out/'refined_dynamics_initial.npz',q=np.stack([x[1] for x in cases]),root=np.stack([x[2] for x in cases]),labels=np.array([x[0] for x in cases]))
 annot.detach();env.close();print('[VALIDATE] COMPLETE',flush=True)
try:main()
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
