"""Check imported USD collision geometry against the saved reset states.
Run with Isaac Sim's pxr library path, or from its Python environment.
"""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from scipy.spatial import ConvexHull
from collision_geometry import CollisionGeometry
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--usd',required=True)
p.add_argument('--output',type=Path,default=Path('outputs/getup_reset_audit'))
a=p.parse_args()
s=json.loads((a.output/'summary.json').read_text());samples=np.load(a.output/'random_samples.npz');nom=np.load(a.output/'nominal.npz')
source=CollisionGeometry('source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf')
root=np.concatenate([nom['root'][:1],samples['root']]);q=np.concatenate([nom['q'][:1],samples['q']])
fk=source.transforms(root[:,:3],root[:,3:7],q,s['joint_names'])
stage=Usd.Stage.Open(a.usd);cache=UsdGeom.XformCache();heights=[];names=[];details=[]
for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
    if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
    body=prim
    while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
    if not body:raise ValueError(str(prim.GetPath()))
    body_name=body.GetName()
    def relative(x):return np.asarray(cache.GetLocalToWorldTransform(x)*cache.GetLocalToWorldTransform(body).GetInverse()).T
    typ=prim.GetTypeName();t=relative(prim)
    total=fk[body_name]@t
    if typ in ('Capsule','Sphere'):
        shape=UsdGeom.Capsule(prim) if typ=='Capsule' else UsdGeom.Sphere(prim)
        radius=shape.GetRadiusAttr().Get()
        scale=np.linalg.norm(total[0,:3,:3],axis=0)
        assert np.allclose(scale,scale[0]),scale
        support=np.full(len(q),radius*scale[0])
        if typ=='Capsule':
            axis='XYZ'.index(str(shape.GetAxisAttr().Get()))
            support+=np.abs(total[:,2,axis])*shape.GetHeightAttr().Get()/2
        z=total[:,2,3]-support
    else:
        points=[]
        for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if child.IsA(UsdGeom.Mesh):
                v=np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get())
                tr=relative(child)
                points.append(v@tr[:3,:3].T+tr[:3,3])
        if not points:raise ValueError((str(prim.GetPath()),typ))
        v=np.concatenate(points);v=v[ConvexHull(v).vertices]
        transform=fk[body_name]
        z=(transform[:,2,:3]@v.T).min(-1)+transform[:,2,3]
    heights.append(z);names.append(str(prim.GetPath()));details.append({'body':body_name,'type':typ})
h=np.stack(heights,-1)
np.save(a.output/'imported_collision_minima.npy',h)
result={'usd':a.usd,'shape_count':len(names),'nominal_min_z':float(h[0].min()),
        'random_min_z':float(h[1:].min()),'random_median_min_z':float(np.median(h[1:].min(-1))),
        'below_minus_1mm_count':int((h[1:].min(-1)<-.001).sum()),
        'worst_shapes':[{'path':names[i],'min_z':float(h[:,i].min()),**details[i]} for i in np.argsort(h.min(0))[:8]],
        'note':'Imported USD geometry incl cylinder-to-capsule conversion; excludes PhysX convex hull cooking tolerances and contact/rest offsets.'}
(a.output/'imported_geometry.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
