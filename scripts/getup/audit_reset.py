"""Audit the actual RL reset without changing training parameters. Still images only."""
import argparse
import json
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('outputs/getup_reset_audit'))
parser.add_argument('--num-envs',type=int,default=16)
parser.add_argument('--samples',type=int,default=1024)
parser.add_argument('--seconds',type=float,default=1.)
parser.add_argument('--no-images',action='store_true')
parser.add_argument('--seed-pose', choices=['supine', 'prone'], default='supine')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not args.no_images:
    args.enable_cameras=True
app=AppLauncher(args).app

import csv
import time
import numpy as np
import torch
from PIL import Image
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import quat_apply
from g1_hoi_learning.assets import G1_INSPIRE_URDF_PATH
from g1_hoi_learning.tasks.getup.env import GetupEnv
from g1_hoi_learning.tasks.getup.env_cfg import GetupEnvCfg
from collision_geometry import CollisionGeometry


def main():
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=GetupEnvCfg()
    cfg.pose.reset_mode=args.seed_pose
    cfg.scene.num_envs=args.num_envs
    cfg.sim.device=args.device
    cfg.pose.randomize_joints=False
    cfg.scene.audit_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=4)
    env=GetupEnv(cfg,render_mode='rgb_array' if not args.no_images else None)
    robot=env.scene['robot']; state=env.getup_state
    ids=torch.arange(env.num_envs,device=env.device)
    geometry=CollisionGeometry(G1_INSPIRE_URDF_PATH)
    all_q=[]; all_root=[]; checks=[]
    summary={'training_config_modified':False, 'joint_names':robot.joint_names,
             'collision_links':[x[0] for x in geometry.shapes],
             'geometry_method':'URDF collision mesh convex support + exact sphere support vs z=0; excludes PhysX inflation/contact offsets',
             'self_collision_enabled':False}
    marker_cfg=SPHERE_MARKER_CFG.replace(prim_path='/Visuals/AuditReferences')
    marker_cfg.markers['sphere'].radius=.015
    markers=VisualizationMarkers(marker_cfg)
    annotator=product=None
    if not args.no_images:
        import omni.replicator.core as rep
        product=rep.create.render_product('/OmniverseKit_Persp',(960,720))
        annotator=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu')
        annotator.attach([product])

    def snapshot():
        root=robot.data.root_state_w.clone()
        root[:,:3]-=env.scene.env_origins
        return root.cpu().numpy(),robot.data.joint_pos.cpu().numpy().copy()

    def picture(name, index=0, view='oblique'):
        if annotator is None: return
        center=env.scene.env_origins[index].cpu().numpy()+np.array([-.15,0,.15])
        eyes={'oblique':np.array([1.4,1.6,1.4]),'side':np.array([0,2.,.12]),'top':np.array([0,.02,2.4])}
        env.sim.set_camera_view(center+eyes[view],center)
        env.sim.forward()
        # Forward updates kinematics without a physics step.
        robot.update(1e-8)
        d=robot.data; torso=state.body_ids['torso_link']
        head=d.body_link_pos_w[:,torso]+quat_apply(d.body_link_quat_w[:,torso],torch.tensor(cfg.pose.head_offset,device=env.device).expand(env.num_envs,-1))
        feet=d.body_link_pos_w[:,[state.body_ids['left_ankle_roll_link'],state.body_ids['right_ankle_roll_link']]]
        markers.visualize(torch.cat([head,feet[:,0],feet[:,1]],dim=0))
        for _ in range(5): env.sim.render()
        rgb=np.asarray(annotator.get_data())
        if rgb.size==0: raise RuntimeError('Empty image after render warmup')
        Image.fromarray(rgb[:,:,:3]).save(args.output/f'{name}.png')

    def validate(label):
        root,q=snapshot()
        limits=robot.data.joint_pos_limits.cpu().numpy()
        mimic_error=0.
        for name,(driver,m,b) in state.action.cfg.mimic.items():
            mimic_error=max(mimic_error,float(np.abs(q[:,state.joint_ids[name]]-q[:,state.joint_ids[driver]]*m-b).max()))
        obs=env.observation_manager.compute()['robot_proprio'].reshape(env.num_envs,3,154)
        expected=torch.tensor(getattr(cfg.pose,args.seed_pose).root_pos,device=env.device)
        root_error=(robot.data.root_pos_w-env.scene.env_origins-expected).abs().max().item()
        expected_quat=np.array(getattr(cfg.pose,args.seed_pose).root_quat); expected_quat/=np.linalg.norm(expected_quat)
        quat_error=float(np.minimum(np.linalg.norm(root[:,3:7]-expected_quat,axis=-1),np.linalg.norm(root[:,3:7]+expected_quat,axis=-1)).max())
        result={'label':label,'finite':bool(np.isfinite(root).all() and np.isfinite(q).all()),
                'joint_limit_violation_rad':float(np.maximum(limits[:,:,0]-q,q-limits[:,:,1]).clip(0).max()),
                'mimic_error_rad':mimic_error,'root_position_error_m':root_error,'root_quaternion_error':quat_error,
                'max_initial_joint_speed':float(robot.data.joint_vel.abs().max()),
                'max_initial_root_speed':float(np.abs(root[:,7:]).max()),
                'history_equal':bool(torch.equal(obs[:,0],obs[:,1]) and torch.equal(obs[:,1],obs[:,2])),
                'last_action_zero':bool((env.action_manager.action==0).all()),
                'success_cleared':bool((state.success_steps==0).all() and not state.success.any())}
        checks.append(result)
        return root,q

    with torch.inference_mode():
        env.reset()
        nominal_root,nominal_q=validate('nominal')
        np.savez(args.output/'nominal.npz',root=nominal_root,q=nominal_q)
        for view in ('oblique','side','top'): picture(f'nominal_{view}',view=view)
        # Verify offline FK against live link-frame poses before trusting geometry screening.
        env.sim.forward(); robot.update(1e-8)
        fk=geometry.transforms(nominal_root[:,:3],nominal_root[:,3:7],nominal_q,robot.joint_names)
        actual=robot.data.body_link_pos_w.cpu().numpy()-env.scene.env_origins.cpu().numpy()[:,None,:]
        summary['fk_max_position_error_m']=max(float(np.abs(fk[n][:,:3,3]-actual[:,i]).max()) for i,n in enumerate(robot.body_names))
        summary['nominal_collision_minima_m']=geometry.minimum_heights(nominal_root[:,:3],nominal_root[:,3:7],nominal_q,robot.joint_names)[0].tolist()
        print('[AUDIT] nominal captured; sampling random resets',flush=True)
        cfg.pose.randomize_joints=True
        torch.manual_seed(20261030)
        for batch in range((args.samples+env.num_envs-1)//env.num_envs):
            env._reset_idx(ids)
            root,q=validate(f'random_batch_{batch}')
            all_root.append(root);all_q.append(q)
            if batch==0:
                for i in range(min(8,env.num_envs)): picture(f'random_{i:02d}',i)
            if batch%16==0: print('[AUDIT] sampled', (batch+1)*env.num_envs,flush=True)
        roots=np.concatenate(all_root)[:args.samples]; qs=np.concatenate(all_q)[:args.samples]
        np.savez_compressed(args.output/'random_samples.npz',root=roots,q=qs)
        minima=geometry.minimum_heights(roots[:,:3],roots[:,3:7],qs,robot.joint_names)
        np.save(args.output/'collision_minima.npy',minima)
        summary['sample_count']=len(qs)
        summary['ground_screen']={'minimum_m':float(minima.min()),'median_minimum_m':float(np.median(minima.min(-1))),
                                 'below_minus_1mm_count':int((minima.min(-1)<-.001).sum()),
                                 'above_plus_3mm_count':int((minima.min(-1)>.003).sum())}
        summary['lowest_links']=[{'link':geometry.shapes[j][0], 'min_z':float(minima[:,j].min()),
                                  'below_minus_1mm_count':int((minima[:,j]<-.001).sum())}
                                 for j in np.argsort(minima.min(0))[:10]]
        for rank,index in enumerate(np.argsort(minima.min(-1))[:3]):
            (args.output/f'lowest_{rank}.json').write_text(json.dumps({'sample_index':int(index),
                'root_pos':roots[index,:3].tolist(),'root_quat_wxyz':roots[index,3:7].tolist(),
                'joint_pos':dict(zip(robot.joint_names,qs[index].tolist())),
                'min_collision_z':float(minima[index].min())},indent=2))
        # Partial reset must not alter another environment's physical state/history/task state.
        state.success_steps[:]=7;state.success[:]=True
        before_root=robot.data.root_state_w.clone();before_q=robot.data.joint_pos.clone()
        before_history=state.history.clone()
        env._reset_idx(ids[::2]);env.observation_manager.compute()
        others=ids[1::2]
        summary['partial_reset']={'root_unchanged':bool(torch.equal(before_root[others],robot.data.root_state_w[others])),
                                  'joints_unchanged':bool(torch.equal(before_q[others],robot.data.joint_pos[others])),
                                  'history_unchanged':bool(torch.equal(before_history[others],state.history[others])),
                                  'success_unchanged':bool((state.success_steps[others]==7).all() and state.success[others].all())}
        (args.output/'reset_checks.json').write_text(json.dumps(checks,indent=2))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('[AUDIT] reset and geometry checks saved',summary['ground_screen'],flush=True)
        # Same random states for both controls, no policy, no teleporting during rollout.
        for randomize in (False,True):
            for mode in ('hold_reset','zero_action'):
                label=('random' if randomize else 'nominal')+'_'+mode
                cfg.pose.randomize_joints=randomize
                torch.manual_seed(1234)
                env.reset()
                initial_root,initial_q=snapshot()
                np.savez(args.output/f'{label}_initial.npz',root=initial_root,q=initial_q)
                action=torch.zeros(env.num_envs,41,device=env.device)
                if mode=='hold_reset': action=robot.data.joint_pos[:,state.drive]-robot.data.default_joint_pos[:,state.drive]
                rows=[];started=time.perf_counter()
                for step in range(round(args.seconds/env.step_dt)):
                    _,_,terminated,truncated,_=env.step(action)
                    contact=env.scene['audit_contacts'].data.net_forces_w_history.norm(dim=-1)
                    force=contact.amax(dim=(1,2))
                    d=robot.data
                    for i in range(env.num_envs):
                        rows.append({'env':i,'time_s':(step+1)*env.step_dt,
                            'pelvis_height':float(d.root_pos_w[i,2]-env.scene.env_origins[i,2]),
                            'base_speed':float(d.root_lin_vel_w[i].norm()),
                            'base_ang_speed':float(d.root_ang_vel_w[i].norm()),
                            'max_joint_speed':float(d.joint_vel[i].abs().max()),
                            'max_body_contact_force':float(force[i]),
                            'terminated':bool(terminated[i]),'truncated':bool(truncated[i])})
                    if step in (0,4,24,49): picture(f'{label}_{step+1:03d}')
                with (args.output/f'{label}.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                summary[label]={'wall_s':time.perf_counter()-started,
                               'max_base_speed':max(r['base_speed'] for r in rows),
                               'max_joint_speed':max(r['max_joint_speed'] for r in rows),
                               'max_contact_force':max(r['max_body_contact_force'] for r in rows),
                               'terminations':sum(r['terminated'] for r in rows),
                               'final_pelvis_heights':[r['pelvis_height'] for r in rows[-env.num_envs:]]}
                (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
                print('[AUDIT]',label,summary[label],flush=True)
    if annotator is not None: annotator.detach()
    env.close()
    print('[AUDIT] COMPLETE',args.output.resolve(),flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        SimulationContext.clear_instance()
        app.close(wait_for_replicator=False)
