"""Offline URDF collision support heights (plane z=0), not PhysX contact offsets."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def origin(node):
    out = np.eye(4)
    if node is not None:
        out[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        out[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return out


class CollisionGeometry:
    def __init__(self, path):
        path = Path(path)
        xml = ET.parse(path).getroot()
        self.joints = list(xml.findall('joint'))
        children = {j.find('child').get('link') for j in self.joints}
        self.root = next(n.get('name') for n in xml.findall('link') if n.get('name') not in children)
        self.shapes = []
        for link in xml.findall('link'):
            for col in link.findall('collision'):
                g = col.find('geometry')
                transform = origin(col.find('origin'))
                mesh, sphere, box, cylinder = (g.find(k) for k in ('mesh','sphere','box','cylinder'))
                radius = 0.
                if mesh is not None:
                    model = trimesh.load(path.parent/mesh.get('filename'), force='mesh')
                    # Linear minima are identical on the original mesh and its convex hull.
                    vertices = model.convex_hull.vertices * np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                elif sphere is not None:
                    vertices = np.zeros((1,3))
                    radius = float(sphere.get('radius'))
                elif box is not None:
                    vertices = trimesh.creation.box(np.fromstring(box.get('size'),sep=' ')).vertices
                elif cylinder is not None:
                    vertices = np.zeros((1,3))
                    radius = (float(cylinder.get('radius')), float(cylinder.get('length'))/2,
                              transform[:3,2].copy())
                else:
                    raise ValueError(ET.tostring(g))
                vertices = vertices @ transform[:3,:3].T + transform[:3,3]
                self.shapes.append((link.get('name'), vertices, radius))

    def transforms(self, root_pos, root_quat_wxyz, q, names):
        n = len(q)
        root = np.broadcast_to(np.eye(4),(n,4,4)).copy()
        root[:,:3,3] = root_pos
        root[:,:3,:3] = Rotation.from_quat(root_quat_wxyz[:,[1,2,3,0]]).as_matrix()
        transforms = {self.root: root}
        indices = dict(zip(names,range(len(names))))
        pending = list(self.joints)
        while pending:
            old_length = len(pending)
            for j in pending[:]:
                parent, child = j.find('parent').get('link'), j.find('child').get('link')
                if parent not in transforms:
                    continue
                local = np.broadcast_to(origin(j.find('origin')),(n,4,4)).copy()
                if j.get('type') != 'fixed':
                    angle = q[:, indices[j.get('name')]]
                    axis = np.fromstring(j.find('axis').get('xyz'),sep=' ')
                    movement = np.broadcast_to(np.eye(4),(n,4,4)).copy()
                    if j.get('type') in ('revolute','continuous'):
                        movement[:,:3,:3] = Rotation.from_rotvec(angle[:,None]*axis).as_matrix()
                    elif j.get('type') == 'prismatic':
                        movement[:,:3,3] = angle[:,None]*axis
                    else:
                        raise ValueError(j.get('type'))
                    local = local @ movement
                transforms[child] = transforms[parent] @ local
                pending.remove(j)
            if len(pending)==old_length:
                raise ValueError('URDF tree did not resolve')
        return transforms

    def minimum_heights(self, root_pos, quat, q, names):
        transforms = self.transforms(root_pos,quat,q,names)
        heights = []
        for link, vertices, radius in self.shapes:
            t = transforms[link]
            support = radius
            if isinstance(radius, tuple):
                r, half_length, axis = radius
                cosine = (t[:,2,:3] @ axis).clip(-1,1)
                support = r*np.sqrt(1-cosine**2)+half_length*np.abs(cosine)
            heights.append((t[:,2,:3] @ vertices.T).min(-1)+t[:,2,3]-support)
        return np.stack(heights,axis=-1)
