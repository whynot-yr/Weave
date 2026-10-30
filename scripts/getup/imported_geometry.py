"""Support heights of imported USD colliders, before PhysX cooking/contact offsets."""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
from collision_geometry import CollisionGeometry

class ImportedGeometry:
    def __init__(self, usd, urdf):
        self.source = CollisionGeometry(urdf)
        stage = Usd.Stage.Open(str(usd))
        cache = UsdGeom.XformCache()
        self.shapes = []
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise ValueError(str(prim.GetPath()))
            def relative(x):
                return np.asarray(cache.GetLocalToWorldTransform(x)*cache.GetLocalToWorldTransform(body).GetInverse()).T
            typ = prim.GetTypeName()
            t = relative(prim)
            if typ in ('Capsule', 'Sphere'):
                shape = UsdGeom.Capsule(prim) if typ == 'Capsule' else UsdGeom.Sphere(prim)
                scale = np.linalg.norm(t[:3, :3], axis=0)
                if not np.allclose(scale, scale[0]):
                    raise ValueError('Nonuniform sphere/capsule scale')
                radius = float(shape.GetRadiusAttr().Get())*scale[0]
                axis = 'XYZ'.index(str(shape.GetAxisAttr().Get())) if typ == 'Capsule' else 0
                half = float(shape.GetHeightAttr().Get())/2 if typ == 'Capsule' else 0.
                data = (t, radius, axis, half)
            else:
                points = []
                for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                    if child.IsA(UsdGeom.Mesh):
                        v = np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get())
                        tr = relative(child)
                        points.append(v@tr[:3,:3].T+tr[:3,3])
                if not points:
                    raise ValueError((str(prim.GetPath()), typ))
                v = np.concatenate(points)
                data = v[ConvexHull(v).vertices]
            self.shapes.append((body.GetName(), str(prim.GetPath()), typ, data))

    def minimum_heights(self, root, q, names):
        fk = self.source.transforms(root[:, :3], root[:, 3:7], q, names)
        heights = []
        for body, path, typ, data in self.shapes:
            transform = fk[body]
            if typ in ('Capsule', 'Sphere'):
                t, radius, axis, half = data
                total = transform@t
                z = total[:,2,3]-radius-np.abs(total[:,2,axis])*half
            else:
                z = (transform[:,2,:3]@data.T).min(-1)+transform[:,2,3]
            heights.append(z)
        return np.stack(heights, axis=-1)

    def lowest_points(self, root, q, names):
        """One supporting point per shape in environment coordinates (not PhysX contacts)."""
        fk = self.source.transforms(root[:, :3], root[:, 3:7], q, names)
        points = []
        for body, path, typ, data in self.shapes:
            transform = fk[body]
            if typ in ('Capsule', 'Sphere'):
                t, radius, axis, half = data
                total = transform @ t
                point = total[:, :3, 3].copy()
                point -= np.sign(total[:, 2, axis])[:, None] * total[:, :3, axis] * half
                point[:, 2] -= radius
            else:
                k = (transform[:, 2, :3] @ data.T).argmin(-1)
                point = np.einsum('nij,nj->ni', transform[:, :3, :3], data[k]) + transform[:, :3, 3]
            points.append(point)
        return np.stack(points, axis=1)
