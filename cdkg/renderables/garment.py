import os
import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.colors as colors
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.meshes import Meshes
from cdkg.configuration import CONFIG as C
from cdkg.renderables.clutches import Clutches
from cdkg.models.cs_tri import CSTri
from cdkg.utils import to_n, to_t, subdivide_meshes
import trimesh

class Garment(Meshes):
    """
    Models a thin layer of flexible cloth as it undergoes deformation on an implicit body surface
    """

    def __init__(self,
                 vertices_ref=None,
                 f_mask=None,
                 att=None,
                 densities=None,
                 body=None,
                 subdivide=False,
                 icon="\u008e",
                 *args,
                 **kwargs
                 ):
        """
        :param f_mask: The faces from the overall body that the garment covers
        :param att: Attachment boolean for every face
        :param densities: Density design variable for every face (over the whole sequence) N, D, F
        :param subdivide: Number of subdivisions from the base SMPL mesh
        :param args: Pass onto parent (meshes)
        :param kwargs: Pass onto parent (meshes)
        :param vertices_ref: The rest state of the garment
        """
        super(Garment, self).__init__(icon=icon, *args, **kwargs)
        self._vertices_ref = vertices_ref
        self.f_mask = f_mask if f_mask is not None else np.ones(self.n_faces, dtype=bool)
        self.att = att if att is not None else np.zeros(self.n_faces, dtype=bool)
        self.densities = densities if densities is not None else np.ones((self.n_frames, self.n_faces))
        self.body = body
        self.subdivide = subdivide

        # GUI Only
        self._pc_scale = True
        self._pc_clip = True
        self._color_strain_multiplier = 0.25
        self._color_mode = 0  # [0] Sensitivities [1] Energy Density Psi, [2] Frobeneus Norm [3] Density,[4] Attachments, [5] Add/Remove Face
        self._strain_line_length = 0.025
        self._smooth = True
        self._show_dense_rest = False
        self._sensitivity_as_bw_map = False
        self._show_avrg_sensitivities = False

        self.gui_modes.update({
            'attach': {'title': ' Attach', 'icon': '\u0081'},
            'density': {'title': ' Density', 'icon': '\u0081'},
            'clutch': {'title': ' Clutch', 'icon': '\u0081'},
        })

        # Material design variables
        self._material_d1 = 0.1
        self._material_d2 = 1.0
        self._material_ym1 = 0.5
        self._material_ym2 = 5.7
        self._material_t1 = 0.00027
        self._material_t2 = 0.00035
        self._material_interpolation = 1.6

        # Energy Quantities
        self.cst_model = self.get_cst_model()
        self.F = None
        self.FT = None
        self.CG = None
        self.E = None
        self.energy_densities = None
        self.energy = None
        self.compression = None
        self.eig_vals = None
        self.eig_vecs = None
        self.sensitivities = None

        self.forward()

        self.face_colors = np.full((self.n_frames, self.n_faces, 4), self.color)

    def forward(self):
        self.forward_geometry()
        self.forward_energy()

    def forward_geometry(self):
        """Set the vertices and faces of the garmnet as a subset of the body mesh"""
        if self.body is None:
            return

        body_v, body_f = self.body.vertices.copy(), self.body.faces.copy()
        if self.subdivide:
            body_v, body_f = subdivide_meshes(body_v, body_f, num_subdivisions=self.subdivide)

        faces_masked = body_f[self.f_mask]

        # Compact vertices
        v_mask = np.unique(faces_masked.reshape(-1))
        self.vertices = np.ascontiguousarray(body_v[:, v_mask])
        self._vertices_ref = self.vertices[0]

        # Compact faces
        v_to_masked_v = np.zeros(body_v.shape[1], dtype=np.int64)
        v_to_masked_v[v_mask] = np.arange(v_mask.shape[0])
        self.faces = v_to_masked_v[faces_masked].astype(np.int32)  # Adjust their indices

        if len(self.densities) != len(self.vertices):
            self.densities = self.densities[[-1]].repeat(len(self.vertices), 0)

        if self.densities.shape[1] != self.faces.shape[0]:
            self.densities = self.densities[:, self.f_mask]

        if self.att.shape[0] != self.faces.shape[0]:
            self.att = self.att[self.f_mask]

        self.update_cst_model()

    def forward_energy(self):
        # Get deformation gradient and principle components from model
        F, FT, CG, E, energy_densities, energy, compression = self.cst_model(
            torch.from_numpy(self.vertices).to(device=C.device, dtype=C.f_precision),
            youngmoduli=torch.from_numpy(self.youngmoduli).to(device=C.device, dtype=C.f_precision),
            thicknesses=torch.from_numpy(self.thicknesses).to(device=C.device, dtype=C.f_precision),
            return_quantities=True
        )

        # Principle components
        eig_vals, eig_vecs = self.cst_model.get_principle_components(CG, eigvecs=True)
        eig_vals, eig_vecs = to_n(eig_vals), to_n(eig_vecs)
        idx = np.abs(eig_vals).argsort(axis=-1)[..., ::-1]
        eig_vals = np.take_along_axis(eig_vals, idx, axis=-1)
        eig_vecs = np.take_along_axis(eig_vecs, idx[:, :, np.newaxis], axis=-1)

        self.F = to_n(F)
        self.FT = to_n(FT)
        self.CG = to_n(CG)
        self.E = to_n(E)
        self.energy_densities = to_n(energy_densities)
        self.energy = to_n(energy)
        self.compression = to_n(compression)
        self.eig_vals = eig_vals
        self.eig_vecs = eig_vecs
        self.sensitivities = self.get_sensitivities()
        self.update_strain_color()
        self.update_strain_lines()


    @property
    def youngmoduli(self):
        youngmoduli = np.zeros_like(self.densities)
        youngmoduli[self.densities == self._material_d1] = self._material_ym1
        youngmoduli[self.densities == self._material_d2] = self._material_ym2
        return youngmoduli

    @property
    def thicknesses(self):
        thicknesses = np.zeros_like(self.densities)
        thicknesses[self.densities == self._material_d1] = self._material_t1
        thicknesses[self.densities == self._material_d2] = self._material_t2
        return thicknesses


    @property
    def f_rest_areas(self):
        return to_n(self.cst_model.f_rest_areas)

    @property
    def area(self):
        return to_n(self.cst_model.f_rest_areas.sum())

    @property
    def dense_faces(self):
        return self.faces[self.densities[self.current_frame_id] > self._material_d1]

    @property
    def area_current(self):
        return (self.f_rest_areas * self.densities[self.current_frame_id].astype(np.int64)).sum(-1)

    @property
    def area_current_percent(self):
        return self.area_current / self.area

    # def faces_masked_to_full(self, f_masked):
    #     return np.arange(self.n_faces)[self.f_mask][f_masked]

    @property
    def vertices_ref(self):
        return self._vertices_ref if self._vertices_ref is not None else self.vertices[0]

    @property
    def color_mode(self):
        return self._color_mode

    @color_mode.setter
    def color_mode(self, color_mode):
        self._color_mode = color_mode
        face_colors = self.get_face_colors()
        self.face_colors = face_colors
        self.redraw()

    @property
    def energy_over_area(self):
        return self.dense_energies / self.dense_areas

    @property
    def dense_areas(self):
        return (self.f_rest_areas * (self.densities > self._material_d1)).sum(-1)

    @property
    def dense_energies(self):
        return (self.energy * (self.densities > self._material_d1)).sum(-1)

    @property
    def clutches(self):
        for n in self.nodes:
            if isinstance(n, Clutches):
                return n
        return None

    def get_cst_model(self):
        vertices_ref = torch.from_numpy(self.vertices_ref).to(device=C.device, dtype=C.f_precision)
        faces = torch.from_numpy(self.faces).to(device=C.device, dtype=C.i_precision)
        return CSTri(faces=faces, vertices_ref=vertices_ref)

    def get_sensitivities(self):
        yms = self.youngmoduli
        s = self.energy_densities.copy()
        ym1_mod, ym2_mod = self.get_sensitivity_mod(yms.max(), yms.min(), p=self._material_interpolation)
        if ym1_mod != 0:
            s[yms == yms.max()] *= ym1_mod
            s[yms == yms.min()] *= ym2_mod

        if not self._smooth: return s

        # Elements that are below the threshold have no influence on the averaging
        areas = self.f_rest_areas[np.newaxis]  # * (youngmoduli / youngmoduli.max())
        faces = self.faces
        # To compute the normals we need to know a mapping from vertex ID to all faces that this vertex is part of.
        # Because we are lazy we abuse trimesh to compute this for us. Not all vertices have the maximum degree, so
        # this array is padded with -1 if necessary.
        vertex_faces = trimesh.Trimesh(self.vertices_ref, self.faces, process=False).vertex_faces

        # Vertex faces indexes last element (-1) since triangles have different # of vertices so we put a 0 there
        s_padded = np.concatenate((s, np.zeros_like(s)[:, [0]]), axis=-1)
        a_padded = np.concatenate((areas, np.zeros_like(areas)[:, [0]]), axis=-1)

        # Diffuse energy to vertices
        vertex_energy = (s_padded[:, vertex_faces] * a_padded[:, vertex_faces]).sum(-1)
        vertex_energy_areas = a_padded[:, vertex_faces].sum(-1)
        vertex_energy_density = vertex_energy / vertex_energy_areas

        # Diffuse back to faces
        return vertex_energy_density[:, faces].mean(-1)

    def get_sensitivity_mod(self, ym1=5.7, ym2=0.5, p=1.6, x=0.001):
        ym1_mod = 1- ym2/ym1
        ym2_mod = ((x**(p-1) * (ym1-ym2) )) / ((x**p)*ym1 + (1-(x**p))*ym2)
        return ym1_mod, ym2_mod

    def get_face_colors(self):
        # Energy Density
        if self._color_mode == 1:
            c = colors.Normalize(vmin=0.0, vmax=0.1 / self._color_strain_multiplier )(self.energy_densities)
            # Shift neutral to blue and invert direction
            f_colors = cm.hsv((2 / 3 - c))
            f_colors[:, :, -1] = self.densities * 8

        # Frobeneus Norm colors
        elif self._color_mode == 2:
            frob_norm_sq = np.power(self.CG - np.eye(2), 2).sum(-1).sum(-1)
            frob_norm_sq = colors.Normalize(vmin=0, vmax=1.0)(frob_norm_sq)
            f_colors = cm.hsv((2 / 3 - frob_norm_sq))

        # Compression
        elif self._color_mode == 3:
            compression_colors = colors.Normalize(vmin=0.5, vmax=1.0, clip=True)(self.compression)
            compression_colors[self.densities < self._material_d2] = 1.0
            f_colors = cm.hsv(2 / 3 - (compression_colors-1))
            f_colors[:, :, -1] = self.densities * 2.5

        # Density colors (B/W)
        elif self._color_mode == 4:
            f_colors = cm.Greys(colors.Normalize()(self.densities))

        # Attachments (B/W)
        elif self._color_mode == 5:
            f_colors = cm.Greys(colors.Normalize()(self.att[np.newaxis].repeat(self.n_frames, axis=0)))

        # Default - Strain coloring according to energy
        else:
            s = self.sensitivities
            if self._show_avrg_sensitivities:
                # s_norm = np.linalg.norm(s[1:], axis=-1)
                s_norm = s[1:].max(-1)

                s = (s[1:] / s_norm[...,np.newaxis]).mean(0)[np.newaxis] * s_norm.mean()
                # s = s[1:].mean(0)[np.newaxis] Not normalized version
                s = s.repeat(len(self.sensitivities), axis=0)
            s = colors.Normalize(vmin=0.0, vmax=0.1 / self._color_strain_multiplier)(s)

            if self._sensitivity_as_bw_map:
                f_colors = cm.Greys(s)
            else:
                # Shift neutral to blue and invert direction
                f_colors = cm.hsv((2 / 3 - s))
                f_colors[:, :, -1] = self.densities * 5
                # f_colors[:, :, 0] += self.att_garment * 2

        return f_colors

    def get_strain_lines(self, eig_vals, eig_vecs, F, scale=True, clip=True):

        # convert eigvals to E quantity
        eig_vals = 0.5 * (eig_vals - 1.0)

        # eig_vecs are already unit length
        if scale:
            eig_vecs = np.abs(eig_vals)[:, :, np.newaxis] * eig_vecs

        if self._show_avrg_sensitivities:
            eig_vecs = eig_vecs.mean(0)[np.newaxis].repeat(len(eig_vecs), axis=0) * 4.0

        # Transform eig_vecs to 3d space
        eig_vecs = np.matmul(F.transpose(0, 1, 3, 2), eig_vecs).transpose((0, 1, 3, 2))

        # Clip of vectors are too long
        if clip:
            e_components_norm = np.linalg.norm(eig_vecs, axis=-1)
            e_clip_mask = e_components_norm > 1.0
            eig_vecs[e_clip_mask] = eig_vecs[e_clip_mask] / e_components_norm[e_clip_mask][..., np.newaxis]

        # Create lines from eig_vecs
        face_centers = np.mean(self.vertices[:, self.faces], axis=-2)[:, :, np.newaxis]
        strain_lines = np.empty((len(self), self.faces.shape[0] * 2, 2, 3), dtype=self.vertices.dtype)
        eig_vec_lines = eig_vecs * self._strain_line_length / 2
        strain_lines[:, 0::2] = face_centers - eig_vec_lines
        strain_lines[:, 1::2] = face_centers + eig_vec_lines

        return strain_lines

    def update_strain_lines(self):
        """Strain lines visualize primary and secondary directions of strain (based on eigvecs)"""
        """We turn these off as they can cause memory issues during TPO"""
        pass
        # c = self.clutches
        # self.strain_lines = self.get_strain_lines(self.eig_vals, self.eig_vecs, self.F)
        # self.nodes = []
        # self.pc_r_main = Lines(self.strain_lines[:, :, 0], r_base=0.00015, mode='lines', color=(1.0, 1.0, 1.0, 1.0),
        #                        name="PC Main")
        # self.pc_r_sec = Lines(self.strain_lines[:, :, 1], r_base=0.00015, mode='lines', color=(1.0, 1.0, 1.0, 1.0),
        #                       name="PC Sec.")
        # self._add_nodes(self.pc_r_main, self.pc_r_sec, enabled=False)
        # if c is not None:
        #     self.add(c)

    def increase_face_density(self, face_ids, brush_size=0):
        # face_id = self.faces_masked_to_full(face_id)
        for n in range(brush_size):
            face_ids = np.unique(self.vertex_faces[self.faces[face_ids]].reshape(-1))[1:]
        self.densities[:, face_ids] *= 10.0
        if np.any(self.densities[:, face_ids] > 1.0):
            self.densities[:, face_ids] = 1.0
        self.update_strain_color()

    def decrease_face_density(self, face_ids, brush_size=0):
        for n in range(brush_size):
            face_ids = np.unique(self.vertex_faces[self.faces[face_ids]].reshape(-1))[1:]

        # face_id = self.faces_masked_to_full(face_id)
        self.densities[:, face_ids] /= 10.0
        if np.any(self.densities[:, face_ids] < 0.1):
            self.densities[:, face_ids] = 0.1
        self.update_strain_color()

    def attach_face(self, face_ids, brush_size=0):
        for n in range(brush_size):
            face_ids = np.unique(self.vertex_faces[self.faces[face_ids]].reshape(-1))[1:]

        # Only attach visible faces
        self.att[face_ids] = True
        self.update_strain_color()

    def detach_face(self, face_ids, brush_size=0):
        for n in range(brush_size):
            face_ids = np.unique(self.vertex_faces[self.faces[face_ids]].reshape(-1))[1:]
        self.att[face_ids] = False
        self.update_strain_color()

    def gui(self, imgui):
        super().gui(imgui)

        # Trigger face color update
        csmu, color_strain_multiplier = imgui.slider_float('Color Strain ##c_multiplier{}'.format(self.unique_name),
                                                               self._color_strain_multiplier, 0.01, 1.0, '%.3f')
        cmu, color_mode = imgui.combo('Color Mode ##color_mode{}'.format(
            self.unique_name), self._color_mode, ['Sensitivities',
                                                  'Energy Density',
                                                  'Frobeneus Norm',
                                                  'Compression',
                                                  'Density',
                                                  'Attachments'])
        su, self._smooth = imgui.checkbox('Smooth Energy ##s_energy_smooth{}'.format(self.unique_name),
                                              self._smooth)
        sbw, self._sensitivity_as_bw_map = imgui.checkbox('BW Sensitivities ##s_energy_smooth{}'.format(self.unique_name),
                                              self._sensitivity_as_bw_map)
        svs, self._show_avrg_sensitivities = imgui.checkbox('Avrg Sensitivities ##s_avrg_sensitivites{}'.format(self.unique_name),
                                              self._show_avrg_sensitivities)

        if su or sbw or svs:
            self.sensitivities = self.get_sensitivities()
            self.update_strain_color()
            self.update_strain_lines()

        if cmu or csmu:
            self._color_strain_multiplier = color_strain_multiplier
            self.color_mode = color_mode

        # Trigger principle component update
        scale_u, self._pc_scale = imgui.checkbox('Scale Components ##pc_scale{}'.format(self.unique_name),
                                                 self._pc_scale)
        clip_u, self._pc_clip = imgui.checkbox('Clip Components ##pc_clip{}'.format(self.unique_name), self._pc_clip)
        if scale_u or clip_u:
            self.strain_lines = self.get_strain_lines(self.eig_vals, self.eig_vecs, self.F, scale=self._pc_scale,
                                                      clip=self._pc_clip)
            self.update_strain_lines()

        # if len(self.vertices) > 1:
        # Energy computation uses a lot of CPU, turned OFF for now.
        #     energy_over_area = self.energy_over_area.astype(np.float32)
        #     imgui.plot_lines("##Energy/Area ", energy_over_area/energy_over_area[0],
        #                      overlay_text='Normalized Energy',
        #                      graph_size=(imgui.get_content_region_available().x, imgui.get_content_region_available().y/4))
            # imgui.plot_lines("Energy ", self.energy[1:].sum(-1).astype(np.float32))
            # imgui.plot_lines("Area ", self.dense_areas[1:].astype(np.float32))

        # imgui.text("Current Energy {:.7f}".format(self.dense_energies[self.current_frame_id] * 1e6))
        imgui.text("Current Area {:.3f}".format(self.area_current))
        imgui.text("Current Area % {:.3f}".format(self.area_current_percent))


    def gui_io(self, imgui):
        if imgui.button("Save"):
            self.to_npz()

        if imgui.button("Save Ref"):
            self.to_npz(save_frames=False)

        if imgui.button("Preview"):
            if not self._show_dense_rest:
                self.densities[0] = self.densities[self.current_frame_id]
            else:
                self.densities[0] = 1.0
            self._show_dense_rest = not self._show_dense_rest
            self.forward_energy()

        if imgui.button("Export Reinforcements OBJ"):
            mesh = trimesh.Trimesh(vertices=self.vertices_ref, faces=self.dense_faces, process=False)
            mesh.export('../data/objs/' + self.name + '_dense.obj')

        if imgui.button("Export Garment OBJ"):
            mesh = trimesh.Trimesh(vertices=self.vertices_ref, faces=self.faces, process=False)
            mesh.export('../data/objs/' + self.name + '.obj')

        if imgui.button("Remove All Material"):
            self.densities[self.densities >= 1.0] = 0.1
            self.forward_energy()

        if imgui.button("Add All Material"):
            self.densities[self.densities < 1.0] = 1.0
            self.forward_energy()

    def gui_context_menu(self, imgui, *menu_position):
       pass

    def update_strain_color(self):
        face_colors = self.get_face_colors()
        self.face_colors = face_colors
        self.redraw()

    def update_cst_model(self):
        self.cst_model = self.get_cst_model()


    @classmethod
    def from_npz(cls, npz_path, body=None, name=None, old_version=False, upsample=False):
        """Load a sequence from an npz file. The filename becomes the name of the sequence"""
        data = np.load(npz_path, allow_pickle=True)
        name = os.path.splitext(os.path.basename(npz_path))[0] if name is None else name
        vertices = data['vertices'] if 'vertices' in data and data['vertices'].size > 1 else None
        vertices_ref = data['vertices_ref'] if 'vertices_ref' in data and data['vertices_ref'].size > 1 else None

        faces = data['faces'] if 'faces' in data and data['faces'].dtype != object else None
        densities = data['densities'] if 'densities' in data and data['densities'].dtype != object else None
        subdivide = data['subdivide'].item() if 'subdivide' in data and data['subdivide'].dtype != object else False
        f_mask = data['f_mask'] if 'f_mask' in data and data['f_mask'].dtype != object else None
        att = data['att'] if 'att' in data and data['att'].dtype != object else None

        if old_version:
            faces_masked = faces[f_mask]
            # Compact faces
            v_to_masked_v = np.zeros(vertices.shape[1], dtype=np.int64)
            # Compact vertices
            v_mask = np.unique(faces_masked.reshape(-1))
            vertices = np.ascontiguousarray(vertices[:, v_mask])
            vertices_ref= np.ascontiguousarray(vertices_ref[v_mask])

            # Compact faces
            v_to_masked_v[v_mask] = np.arange(v_mask.shape[0])
            faces = v_to_masked_v[faces_masked]  # Adjust their indices
            densities = densities[:, f_mask]
            att = att[f_mask]

        if upsample:
            # Increase resolution of garment mesh
            densities = np.tile(densities, 4)
            att = np.tile(att, 4)
            f_mask = np.tile(f_mask, 4)
            subdivide += 1
            vertices, faces = subdivide_meshes(vertices, faces, num_subdivisions=subdivide)
            vertices_ref = vertices[0]

        return cls(
            vertices_ref=vertices_ref,
            vertices=vertices,
            faces=faces,
            densities=densities,
            att=att,
            f_mask=f_mask,
            body=body,
            subdivide=subdivide,
            name=name
        )

        # EG 22 garments have a slightly different definition
        # if old_version:
        # garment.densities = garment.densities[:, f_mask]
        # garment.att = garment.att[f_mask]
        # garment.forward_geometry()
        # return garment

    def to_npz(self, path=os.path.join(C.datasets.kg, "garments/"), save_frames=True):
        """Save garments to custom NPZ format"""

        v = self.vertices if save_frames else self.vertices[[0]]
        d = self.densities if save_frames else self.densities[[0]]
        np.savez_compressed(path + self.name + '.npz',
                            vertices=v.astype(np.float32),
                            faces=self.faces,
                            densities=d,
                            att=self.att,
                            f_mask=self.f_mask,
                            subdivide=self.subdivide,
                            vertices_ref=self.vertices_ref
                            )

    def make_renderable(self, ctx):
        super().make_renderable(ctx)
        self.update_strain_color()
        self.update_strain_lines()

    def process_mmi(self, mmi, button, _rmb, _lmb, _brush_size):
        """Processes Mesh-Mouse intersection (Direct interaction with mesh)"""

        # Attach / Detach parts of the garment
        if self.selected_mode in 'attach':
            if button == _lmb:
                self.attach_face(mmi.tri_id)
            elif button == _rmb:
                self.detach_face(mmi.tri_id)

        # Stiffen / Soften parts of the garment
        if self.selected_mode in 'density':
            if button == _lmb:
                self.increase_face_density(mmi.tri_id, _brush_size)
            elif button == _rmb:
                self.decrease_face_density(mmi.tri_id, _brush_size)

        # Place and Edit clutches
        if self.selected_mode in 'clutch':

            # Did we click on existing clutch?
            nearby_clutch_id = None
            if self.clutches is not None:
                nearby_clutch_id = self.clutches.nearest_clutch(mmi.point_local)

            # Remove
            if button == _rmb and nearby_clutch_id is not None:
                self.clutches.remove_clutch(nearby_clutch_id)

            # Left clicked not near any clutches - add clutch
            if button == _lmb and nearby_clutch_id is None:
                if self.clutches is None:
                    # Make a new clutch
                    clutch = Clutches(bc_coords=np.array([mmi.bc_coords]),
                                      bc_fids=np.array([mmi.tri_id]),
                                      surface_meshes=self)
                    self.add(clutch)
                else:
                    # Add to existing clutches
                    self.clutches.add_clutch(bc_coord=mmi.bc_coords, bc_fid=mmi.tri_id)

    @property
    def selected_mode(self):
        return self._selected_mode

    @selected_mode.setter
    def selected_mode(self, selected_mode):
        """Make sure to change the color mode rendering when changing gui modes"""
        self._selected_mode = selected_mode
        if selected_mode == 'view' or selected_mode == 'clutch':
            self.color_mode = 0
        elif selected_mode == 'density':
            self.color_mode = 4
        elif selected_mode == 'attach':
            self.color_mode = 5
