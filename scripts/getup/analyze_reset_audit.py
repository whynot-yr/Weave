"""Summarize recorded reset samples; plot URDF collision projections, not visual meshes."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collision_geometry import CollisionGeometry

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('outputs/getup_reset_audit'))
args=parser.parse_args()
p=args.output
s=json.loads((p/'summary.json').read_text())
g=CollisionGeometry('source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf')
nom=np.load(p/'nominal.npz'); rnd=np.load(p/'random_samples.npz'); mins=np.load(p/'collision_minima.npy')
worst=int(mins.min(-1).argmin())
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
for ax,label,root,q in [(axes[0],'Nominal reset',nom['root'][:1],nom['q'][:1]),
                         (axes[1],f'Worst sampled reset #{worst}',rnd['root'][worst:worst+1],rnd['q'][worst:worst+1])]:
    tf=g.transforms(root[:,:3],root[:,3:7],q,s['joint_names'])
    for link,v,r in g.shapes:
        t=tf[link][0]
        v=v@t[:3,:3].T+t[:3,3]
        if isinstance(r,tuple):
            radius, half_length, local_axis = r
            axis = t[:3,:3] @ local_axis
            basis = np.eye(3)[np.argmin(np.abs(axis))]
            e1 = np.cross(axis,basis); e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis,e1)
            theta = np.linspace(0,2*np.pi,128)
            circle = radius*(np.cos(theta)[:,None]*e1+np.sin(theta)[:,None]*e2)
            surface = np.concatenate([v[0]+circle-half_length*axis,v[0]+circle+half_length*axis])
            xz=surface[:,[0,2]];xz=xz[ConvexHull(xz).vertices]
        elif r>0:
            theta=np.linspace(0,2*np.pi,32)
            xz=v[0,[0,2]]+r*np.stack([np.cos(theta),np.sin(theta)],axis=-1)
        elif len(v)>=3:
            xz=v[:,[0,2]]
            try: xz=xz[ConvexHull(xz).vertices]
            except Exception: continue
        else:
            continue
        ax.fill(xz[:,0],xz[:,1],color='tomato' if xz[:,1].min()<-.001 else 'steelblue',alpha=.35,linewidth=.4,edgecolor='black')
    ax.axhline(0,color='black',linewidth=1)
    ax.axhspan(-.15,0,color='gray',alpha=.12)
    ax.set_title(label);ax.set_ylabel('world z (m)');ax.set_ylim(-.15,.5);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
axes[1].set_xlabel('world x (m); side projection, all y combined')
fig.suptitle('Source collision geometry vs ground z=0 (red: extends below -1 mm)')
fig.tight_layout();fig.savefig(p/'collision_side_projection.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
ax.hist(mins.min(-1)*100,bins=40,color='steelblue');ax.axvline(0,color='black')
ax.set_xlabel('Lowest source collision point (cm)');ax.set_ylabel('Reset count');fig.tight_layout()
fig.savefig(p/'ground_clearance_histogram.png',dpi=150);plt.close(fig)
nom_min=min(s['nominal_collision_minima_m'])
calc={'nominal_min_z_m':nom_min,'sample_min_z_quantiles_m':dict(zip(['min','p05','median','p95','max'],np.quantile(mins.min(-1),[0,.05,.5,.95,1]).tolist())),
      'nominal_root_height_for_2mm_clearance_candidate':.06-nom_min+.002,
      'per_sample_root_height_for_2mm_clearance_quantiles':np.quantile(.06-mins.min(-1)+.002,[0,.5,.95,1]).tolist(),
      'note':'Height estimates only; no training/reset parameters were changed. Excludes PhysX contact offsets and cylinder-to-capsule conversion.'}
(p/'geometry_analysis.json').write_text(json.dumps(calc,indent=2))
print(json.dumps(calc,indent=2))
