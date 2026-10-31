"""Actual-environment checks for named and mixed seeds, histories, geometry and attribution."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/getup_mixed_reset_check'));AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
from isaaclab.sim import SimulationContext
from g1_hoi_learning.tasks.getup.env import GetupEnv
from g1_hoi_learning.tasks.getup.env_cfg import GetupEnvCfg
from g1_hoi_learning.assets import G1_INSPIRE_URDF_PATH
from imported_geometry import ImportedGeometry

def main():
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cfg=GetupEnvCfg();cfg.scene.num_envs=16;cfg.sim.device=a.device;env=GetupEnv(cfg);env.reset();r=env.scene['robot'];s=env.getup_state;ids=torch.arange(16,device=env.device);default=r.data.default_joint_pos.clone()
 geom=ImportedGeometry('/tmp/IsaacLab/usd_20261030_215206_1884/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd',G1_INSPIRE_URDF_PATH)
 result={};grav=[]
 def check():
  obs=env.observation_manager.compute()['robot_proprio'].reshape(16,3,154)
  assert torch.equal(obs[:,0],obs[:,1]) and torch.equal(obs[:,1],obs[:,2])
  assert (env.action_manager.action==0).all() and (r.data.joint_vel==0).all()
  assert (r.data.root_state_w[:,7:]==0).all()
  assert (s.success_steps==0).all() and not s.success.any()
  torch.testing.assert_close(r.data.root_pos_w-env.scene.env_origins,s.seed_root_pos[s.episode_seed_id])
  torch.testing.assert_close(r.data.root_quat_w,s.seed_root_quat[s.episode_seed_id])
  limits=r.data.joint_pos_limits
  assert ((r.data.joint_pos>=limits[...,0]-1e-6)&(r.data.joint_pos<=limits[...,1]+1e-6)).all()
  for n,(d,m,b) in s.action.cfg.mimic.items():torch.testing.assert_close(r.data.joint_pos[:,s.joint_ids[n]],r.data.joint_pos[:,s.joint_ids[d]]*m+b)
  torch.testing.assert_close(r.data.default_joint_pos,default)
  torch.testing.assert_close(s.action._offset,default[:,s.drive])
  root=r.data.root_state_w.cpu().numpy().copy();root[:,:3]-=env.scene.env_origins.cpu().numpy();return root,r.data.joint_pos.cpu().numpy().copy()
 for seed_id,name in enumerate(s.seed_names):
  cfg.pose.reset_mode=name;cfg.pose.randomize_joints=False;env._reset_idx(ids);rt,q=check();grav.append(r.data.projected_gravity_b[0].clone())
  artifact='outputs/getup_supine_exploration/refined_fresh.json' if name=='supine' else 'outputs/getup_prone_refined/refined_arms.json'
  expected=json.loads(Path(artifact).read_text());np.testing.assert_allclose(q,np.tile([expected['joint_pos'][n] for n in r.joint_names],(16,1)),atol=1e-6)
  assert (s.episode_seed_id==seed_id).all()
  cfg.pose.randomize_joints=True;torch.manual_seed(2026110105+seed_id);roots=[];qs=[]
  for _ in range(64):env._reset_idx(ids);rt,q=check();roots.append(rt);qs.append(q)
  roots=np.concatenate(roots);qs=np.concatenate(qs);h=geom.minimum_heights(roots,qs,r.joint_names)
  np.savez_compressed(a.output/(name+'.npz'),root=roots,q=qs,minima=h)
  result[name]={'samples':len(qs),'no_penetration':int((h.min(-1)>=0).sum()),'min_collision_z':float(h.min()),'state_checks_passed':True}
  print('[MIXED CHECK]',name,result[name],flush=True)
 assert grav[0][0]*grav[1][0]<0
 cfg.pose.reset_mode='mixed';counts=torch.zeros(2,dtype=torch.long,device=env.device);changed=False;last=s.episode_seed_id.clone()
 torch.manual_seed(2026110200)
 for _ in range(128):
  env._reset_idx(ids);check();counts+=torch.bincount(s.episode_seed_id,minlength=2);changed|=bool((last!=s.episode_seed_id).any());last=s.episode_seed_id.clone()
 assert changed and (counts>800).all() and (counts<1250).all()
 result['mixed_counts']=counts.tolist()
 # Partial resets preserve all untouched histories, physical state and seed attribution.
 before=(r.data.root_state_w.clone(),r.data.joint_pos.clone(),s.history.clone(),s.episode_seed_id.clone())
 env._reset_idx(ids[::2]);env.observation_manager.compute()
 for old,new in zip(before,(r.data.root_state_w,r.data.joint_pos,s.history,s.episode_seed_id)):assert torch.equal(old[1::2],new[1::2])
 # Repeating the same RNG seed produces exactly the same seed selection and joint randomization.
 torch.manual_seed(73);env._reset_idx(ids);rt1,q1=check();id1=s.episode_seed_id.clone()
 torch.manual_seed(73);env._reset_idx(ids);rt2,q2=check();assert np.array_equal(q1,q2) and np.array_equal(rt1,rt2) and torch.equal(id1,s.episode_seed_id)
 # Completion statistics must belong to the episode that ended, not its newly sampled seed.
 for i,name in enumerate(s.seed_names):cfg.pose.reset_mode=name;env._reset_idx(ids[i:i+1])
 cfg.pose.reset_mode='mixed';env.pop_seed_statistics()
 env.reset_terminated=torch.zeros(16,dtype=torch.bool,device=env.device)
 env.reset_time_outs=torch.zeros_like(env.reset_terminated)
 env.episode_length_buf[:2]=torch.tensor([3,7],device=env.device);env.reset_time_outs[:2]=True
 s.success[:2]=torch.tensor([True,False],device=env.device);s.success_steps[:2]=torch.tensor([s.required_steps,0],device=env.device)
 env.reward_manager.group_sums[:2]=torch.tensor([[1.,2.,3.,4.],[10.,20.,30.,40.]],device=env.device)
 env._reset_idx(ids[:2]);stats=env.pop_seed_statistics()
 torch.testing.assert_close(stats[:,1:5],torch.tensor([[1.,1.,1.,3.],[1.,0.,0.,7.]],dtype=stats.dtype,device=env.device))
 torch.testing.assert_close(stats[:,5:],torch.tensor([[1.,2.,3.,4.],[10.,20.,30.,40.]],dtype=stats.dtype,device=env.device))
 env._reset_idx(ids[:2]);assert env.pop_seed_statistics()[:,1:].count_nonzero()==0
 result.update(partial_reset_passed=True,reproducibility_passed=True,old_seed_attribution_passed=True,observations=462)
 (a.output/'result.json').write_text(json.dumps(result,indent=2));env.close();print('[MIXED CHECK] PASSED',result,flush=True)
try:main()
except Exception:
 import traceback
 traceback.print_exc()
 raise
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
