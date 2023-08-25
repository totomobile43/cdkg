import os
import numpy as np
import torch
from cdkg.configuration import CONFIG as C
from aitviewer.renderables.star import STARSequence
from cdkg.utils import subdivide_meshes
from cdkg.renderables.garment import Garment
from aitviewer.utils import to_torch, to_numpy
from aitviewer.models.star import STARLayer
from aitviewer.utils.decorators import hooked

class Body(STARSequence):
    """
    Represents a temporal sequence of SMPL poses. Can be loaded from disk or initialized from memory.
    """

    def __init__(self,
                 poses_body,
                 smpl_layer,
                 poses_root=None,
                 betas=None,
                 trans=None,
                 device=C.device,
                 dtype=C.f_precision,
                 include_root=True,
                 normalize_root=False,
                 is_rigged=True,
                 show_joint_angles=False,
                 **kwargs):
        """
        Initializer.
        :param poses_body: An array (numpy ar pytorch) of shape (F, N_JOINTS*3) containing the pose parameters of the
          body, i.e. without hands or face parameters.
        :param smpl_layer: The SMPL layer that maps parameters to joint positions and/or dense surfaces.
        :param poses_root: An array (numpy or pytorch) of shape (F, 3) containing the global root orientation.
        :param betas: An array (numpy or pytorch) of shape (N_BETAS, ) containing the shape parameters.
        :param trans: An array (numpy or pytorch) of shape (F, 3) containing the global root translation.
        :param device: The pytorch device for computations.
        :param dtype: The pytorch data type.
        :param include_root: Whether or not to include root information. If False, no root translation and no root
          rotation is applied.
        :param normalize_root: Whether or not to normalize the root. If True, the global root translation in the first
          frame is zero and the global root orientation is the identity.
        :param is_rigged: Whether or not to display the joints as a skeleton.
        :param show_joint_angles: Whether or not the coordinate frames at the joints should be visualized.
        :param kwargs: Remaining arguments for rendering.
        """
        assert len(poses_body.shape) == 2

        super(Body, self).__init__(poses_body, smpl_layer, poses_root, betas, trans, device=device, dtype=dtype,
                                           include_root=include_root, normalize_root=normalize_root,
                                           is_rigged=is_rigged, show_joint_angles=show_joint_angles, **kwargs)

        self.mesh_seq.name = "body_mesh"
        self.gui_modes.update({
            'edit_garment': {'title': ' Edit Garment', 'icon': '\u0081'},
        })

    @property
    def poses(self):
        return torch.cat((self.poses_root, self.poses_body), dim=1)

    def to_npz(self, path=''):
        np.savez_compressed(path + self.name + '.npz',
                            poses_body=self.poses_body.detach().cpu().numpy(),
                            poses_root=self.poses_root.detach().cpu().numpy(),
                            betas=self.betas.detach().cpu().numpy(),
                            trans=self.trans.detach().cpu().numpy())

    @classmethod
    def from_npz(cls, npz_path, smpl_layer, start_frame=None, end_frame=None, step_frames=1, **kwargs):
        """Load a skeleton sequence from an npz file. The filename becomes the name of the sequence"""
        data = np.load(npz_path, allow_pickle=True)
        name = os.path.splitext(os.path.basename(npz_path))[0]

        if 'poses' in data:
            poses_body = data['poses'][:, 3:smpl_layer.n_joints_total * 3]
            poses_root = data['poses'][:, 0:3]
        else:
            poses_body = data['poses_body']
            poses_root = data['poses_root']

        sf = start_frame or 0
        ef = end_frame or len(poses_body)
        frames = slice(sf, ef, step_frames)

        body = cls(
            poses_body=poses_body[frames],
            poses_root=poses_root[frames],
            smpl_layer=smpl_layer,
            betas=data['betas'],
            # trans=data['trans'][frames],
            name=name,
            **kwargs)

        return body

    @classmethod
    def t_pose(cls, model=None, betas=None, frames=1, **kwargs):
        """Creates a SMPL sequence whose single frame is a SMPL mesh in T-Pose."""

        if model is  None:
            model = STARLayer(device=C.device)

        poses_body = np.zeros([frames, model.n_joints_body * 3])
        poses_root = np.zeros([frames, 3])
        return cls(poses_body=poses_body, smpl_layer=model, poses_root=poses_root, betas=betas, **kwargs)

    @classmethod
    def a_pose(cls, model=None, betas=None, frames=1, **kwargs):
        """Creates a SMPL sequence whose single frame is a SMPL mesh in T-Pose."""
        t_pose = Body.t_pose(model, betas, frames, **kwargs)
        # Arm down
        t_pose.poses_body[:, 47] -= 0.3
        t_pose.poses_body[:, 50] += 0.3
        t_pose.poses_body[:, 38] -= 0.4
        t_pose.poses_body[:, 41] += 0.4

        # Arm Forward
        t_pose.poses_body[:, 46] -= 0.1
        t_pose.poses_body[:, 49] += 0.1
        t_pose.poses_body[:, 37] -= 0.1
        t_pose.poses_body[:, 40] += 0.1

        # Elbow slight bend
        t_pose.poses_body[:, 52] -= 0.5
        t_pose.poses_body[:, 53] += 0.1
        t_pose.poses_body[:, 55] += 0.5
        t_pose.poses_body[:, 56] -= 0.1
        # t_pose.poses_body[:, 33] += 0.088

        # Legs not touching
        t_pose.poses_body[:, 2] += 0.1
        t_pose.poses_body[:, 5] -= 0.1
        t_pose.poses_body[:, 11] -= 0.08
        t_pose.poses_body[:, 14] += 0.08
        t_pose.poses_body[:, 19] += 0.15
        t_pose.poses_body[:, 22] -= 0.15

        return cls(poses_body=t_pose.poses_body, poses_root=t_pose.poses_root, smpl_layer=t_pose.smpl_layer, betas=betas)

    @hooked
    def gui_io(self, imgui):
        if imgui.button("Export Garment from f_mask (.npz)"):
            g = Garment(vertices=self.vertices, faces=self.faces, f_mask=self.f_mask, subdivide=0, name=self.garment.name+"_new")
            g.to_npz(save_frames=False)

    @property
    def garment(self):
        for n in self.nodes:
            if isinstance(n, Garment):
                return n
        return None

    def add_garment(self, garment):
        self.add(garment)
        self.n_frames = max(self.n_frames, garment.n_frames)
        self.mold_to_garment(garment)

    def mold_to_garment(self, garment, indent=0.002):

        # Get v-mask
        body_v, body_f = self.vertices.copy(), self.faces.copy()
        if garment.subdivide:
            body_v, body_f = subdivide_meshes(body_v, body_f, num_subdivisions=garment.subdivide)
        faces_masked = body_f[garment.f_mask]
        v_mask = np.unique(faces_masked.reshape(-1))

        n_vertices = self.vertices.shape[1]
        n_frames = garment.vertices.shape[0]
        v = self.vertices.copy()
        vn = self.vertex_normals
        # Repeat last frame of vertex normals of body until it fits lengths of vertices in garment
        if len(v) < n_frames:
            v = np.concatenate((v, v[[-1]].repeat(n_frames-len(v),0)))
            vn = np.concatenate((vn, vn[[-1]].repeat(n_frames-len(vn),0)))
        rvmask = v_mask[v_mask < n_vertices]
        v[:, rvmask] -= vn[:, rvmask] * indent

        self.vertices = v[:len(self.mesh_seq.vertices)]
        self.mesh_seq.vertices = self.vertices
        self.mesh_seq.redraw()

    def process_mmi(self, mmi, button, _rmb, _lmb, _brush_size):
        # Mask / Unmask faces to add remove material from garment
        if self.selected_mode in 'edit_garment':
            if button == _lmb:
                self.f_mask[mmi.tri_id] = True
            elif button == _rmb:
                self.f_mask[mmi.tri_id] = False
            self.color_mode = 1

    def key_event(self, key, wnd_keys):
        if key == wnd_keys.V:
            self.selected_mode = 'view'
            self.color_mode = 0

        elif key == wnd_keys.M:
            self.selected_mode = 'edit_garment'
            self.color_mode = 1
