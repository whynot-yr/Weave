"""Paired per-joint/group reset sensitivity at a diagnostic root height."""
import argparse, json, csv, time
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('outputs/getup_reset_sensitivity_074'))
p.add_argument('--usd',default='/tmp/IsaacLab/usd_20261030_215206_1884/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd')
p.add_argument('--root-height',type=float,default=.074)
p.add_argument('--samples',type=int,default=1024)
p.add_argument('--scan-points',type=int,default=101)
p.add_argument('--seed',type=int,default=20261030)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(); a.enable_cameras=True
app=AppLauncher(a).app
import numpy as np
import torch
from PIL import Image
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from g1_hoi_learning.tasks.getup.env import GetupEnv
from g1_hoi_learning.tasks.getup.env_cfg import GetupEnvCfg
from g1_hoi_learning.assets import G1_INSPIRE_URDF_PATH
from imported_geometry import ImportedGeometry

def main():
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg=GetupEnvCfg();cfg.scene.num_envs=16;cfg.sim.device=a.device
    cfg.pose.root_pos=(0.,0.,a.root_height);cfg.pose.randomize_joints=False
    cfg.scene.audit_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=4)
    env=GetupEnv(cfg);env.reset();s=env.getup_state;r=s.robot;names=r.joint_names
    ids=torch.arange(env.num_envs,device=env.device)
    root=r.data.root_state_w[0].cpu().numpy().copy();root[:3]-=env.scene.env_origins[0].cpu().numpy()
    nominal=r.data.joint_pos[0].cpu().numpy().copy()
    geom=ImportedGeometry(a.usd,G1_INSPIRE_URDF_PATH)
    # Check extracted geometry matches the previous independent USD checker.
    old=np.load('outputs/getup_reset_audit/random_samples.npz')
    prev=np.load('outputs/getup_reset_audit/imported_collision_minima.npy')[1:]
    regression=float(np.max(np.abs(geom.minimum_heights(old['root'],old['q'],names)-prev)))
    assert regression<1e-7,regression
    drive=list(s.drive);mimic=s.action.cfg.mimic
    limits=r.data.joint_pos_limits[0].cpu().numpy().copy()
    for name,(driver,m,b) in mimic.items():
        j,d=names.index(name),names.index(driver);bounds=(limits[j]-b)/m
        limits[d,0]=max(limits[d,0],bounds.min());limits[d,1]=min(limits[d,1],bounds.max())
    def derive(q):
        for name,(driver,m,b) in mimic.items():q[:,names.index(name)]=q[:,names.index(driver)]*m+b
        return q
    # Use actual reset, not a parallel implementation, to generate the shared table.
    cfg.pose.randomize_joints=True;torch.manual_seed(a.seed)
    full=[]
    for batch in range((a.samples+15)//16):
        env._reset_idx(ids);full.append(r.data.joint_pos.cpu().numpy().copy())
    full=np.concatenate(full)[:a.samples]
    np.savez_compressed(a.output/'inputs.npz',root=root,nominal=nominal,full=full,limits=limits,drive=drive)
    feet={side:np.array([i for i,x in enumerate(geom.shapes) if x[0]==side+'_ankle_roll_link']) for side in ('left','right')}
    assert all(len(x)>0 for x in feet.values())
    base=geom.minimum_heights(root[None],nominal[None],names)[0]
    records=[];worst=[]
    def evaluate(label,q,kind):
        h=geom.minimum_heights(np.repeat(root[None],len(q),0),q,names)
        z=h.min(-1);idx=int(z.argmin());shape=int(h[idx].argmin())
        item={'label':label,'kind':kind,'n':len(q),'min_z':float(z.min()),'p05_z':float(np.quantile(z,.05)),
              'median_z':float(np.median(z)),'below_0':int((z<0).sum()),'below_minus_1mm':int((z<-.001).sum()),
              'delta_min_from_nominal':float(z.min()-base.min()),'worst_index':idx,
              'worst_body':geom.shapes[shape][0],'worst_shape':geom.shapes[shape][1]}
        for side,fi in feet.items():
            f=h[:,fi].min(-1);item[side+'_foot_min']=float(f.min());item[side+'_foot_below_minus_1mm']=int((f<-.001).sum())
        np.savez_compressed(a.output/(label+'.npz'),q=q,minima=h)
        records.append(item)
        worst.append((float(z.min()),label,q[idx].copy()))
        return item
    def subset(indices):
        q=np.repeat(nominal[None],len(full),0);q[:,indices]=full[:,indices];return derive(q)
    evaluate('baseline',nominal[None],'baseline')
    full_result=evaluate('E_all',full,'group')
    print('[SENS] baseline/full',float(base.min()),full_result,flush=True)
    for j in drive:
        evaluate('single_random_'+names[j],subset([j]),'single_random')
        lo=max(limits[j,0],nominal[j]-.1*abs(nominal[j])-.1)
        hi=min(limits[j,1],nominal[j]+.1*abs(nominal[j])+.1)
        q=np.repeat(nominal[None],a.scan_points,0);q[:,j]=np.linspace(lo,hi,a.scan_points)
        evaluate('scan_'+names[j],derive(q),'scan')
    print('[SENS] all 41 single-joint tests complete',flush=True)
    for side in ('left','right','both'):
        sides=['left','right'] if side=='both' else [side]
        selections={'A_hip_pitch':['hip_pitch'],'B_knee':['knee'],'C_ankle_pitch':['ankle_pitch'],
                    'C_ankle_roll':['ankle_roll'],'C_ankle_both':['ankle_pitch','ankle_roll'],
                    'D_pitch_chain':['hip_pitch','knee','ankle_pitch','ankle_roll'],
                    'D_full_leg':['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']}
        for label,parts in selections.items():
            js=[names.index(side_name+'_'+part+'_joint') for side_name in sides for part in parts]
            evaluate(label+'_'+side,subset(js),'group')
    for j in drive:
        q=full.copy();q[:,j]=nominal[j]
        evaluate('restore_'+names[j],derive(q),'restore')
    with (a.output/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    metadata={'root_height':a.root_height,'seed':a.seed,'names':names,'drive_names':[names[j] for j in drive],
              'geometry_regression_max_error':regression,'shape_count':len(geom.shapes),
              'shapes':[{'body':x[0],'path':x[1],'type':x[2]} for x in geom.shapes],
              'training_parameters_modified':False,'metrics':records}
    (a.output/'results.json').write_text(json.dumps(metadata,indent=2))
    print('[SENS] static tests saved',len(records),flush=True)
    # Most severe distinct tested poses + the baseline; fixed targets, no policy.
    selected=[('baseline',nominal)]
    for z,label,q in sorted(worst,key=lambda x:x[0]):
        if not any(np.allclose(q,other,atol=1e-7) for _,other in selected):selected.append((label,q))
        if len(selected)==16:break
    q=np.stack([x[1] for x in selected]);qt=torch.tensor(q,device=env.device,dtype=r.data.joint_pos.dtype)
    cfg.pose.randomize_joints=False;env.reset()
    r.write_joint_state_to_sim(qt,torch.zeros_like(qt),env_ids=ids);r.set_joint_position_target(qt)
    action=qt[:,s.drive]-r.data.default_joint_pos[:,s.drive]
    import omni.replicator.core as rep
    product=rep.create.render_product('/OmniverseKit_Persp',(960,720));annot=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');annot.attach([product])
    def picture(index,label):
        center=env.scene.env_origins[index].cpu().numpy()+np.array([-.15,0,.15])
        env.sim.set_camera_view(center+np.array([0,2,.12]),center)
        env.sim.forward();r.update(1e-8)
        for _ in range(5):env.sim.render()
        Image.fromarray(np.asarray(annot.get_data())[:,:,:3]).save(a.output/(label+'.png'))
    picture(0,'baseline_side');picture(1,'worst_side')
    rows=[]
    for step in range(50):
        _,_,term,trunc,_=env.step(action)
        force=env.scene['audit_contacts'].data.net_forces_w_history.norm(dim=-1).amax(dim=(1,2))
        for i,(label,_) in enumerate(selected):
            rows.append({'case':i,'label':label,'step':step+1,'base_speed':float(r.data.root_lin_vel_w[i].norm()),
                'joint_speed':float(r.data.joint_vel[i].abs().max()),'pelvis_height':float(r.data.root_pos_w[i,2]-env.scene.env_origins[i,2]),
                'contact_force':float(force[i]),'terminated':bool(term[i]),'truncated':bool(trunc[i])})
    picture(0,'baseline_after_1s');picture(1,'worst_after_1s')
    with (a.output/'dynamics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(a.output/'dynamics_initial.npz',root=root,q=q,labels=np.array([x[0] for x in selected]))
    annot.detach();env.close()
    print('[SENS] COMPLETE',a.output,flush=True)
try:main()
finally:
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)
