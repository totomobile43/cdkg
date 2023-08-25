import numpy as np
import torch
import os
import matplotlib.cm as cm
import matplotlib.colors as colors
from trimesh.triangles import barycentric_to_points, points_to_barycentric
from aitviewer.scene.node import Node
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.meshes import Meshes
from cdkg.configuration import CONFIG as C
from cdkg.models.cs_tri import CSTri
from cdkg.utils import to_n, to_t, clutch_model, barycentric_explicit_to_implicit

class Clutches(Node):
    """
    Models a clutch attached to the surface of a mesh (over time)
    Clutches are embedded on a given surface mesh and are parametrically generated
    from a barycentric coordinate/direction and length/width
    """

    def __init__(self,
                 bc_coords=None,
                 bc_fids=None,
                 bc_dirs=np.array([0.5, 0.5])[np.newaxis],
                 lengths=np.array([0.15]),
                 widths=np.array([0.01]),
                 pre_strains=np.array([1.0]),
                 states=None,
                 surface_meshes=None,
                 vertices=None,
                 faces=None,
                 vertices_ref=None,
                 icon="\u0095",
                 *args,
                 **kwargs):
        """
        :param bc_coords: (implicit) Barycentric coordinates for the center point of each clutch C x 2
        :param bc_fids: Face id of each clutch C x 1
        :param bc_dirs: Direction that each clutch is oriented in (implicit) barycentric coords D x 2
        :param lengths: Length of each clutch C x L
        :param widths: Length of each clutch C x W
        :param pre_strains: Apply a pre-strain amount to each clutch (0.0 to 1.0 for no pre-strain) C x P
        :param states: The states of each clutch for each frame - i.e. is it ON / OFF / another state
        :param surface_meshes: A meshes objects containing vertices/faces that clutches are embedded on
        :param vertices: Length of each clutch C x W
        :param faces: Length of each clutch C x W
        :param args: Pass onto parent (meshes)
        :param kwargs: Pass onto parent (meshes)
        """
        super(Clutches, self).__init__(icon=icon, gui_material=False, *args, **kwargs)
        assert surface_meshes is not None or vertices is not None, "Require at least a surface mesh or opt. vertices"
        self.n_frames = len(vertices) if vertices is not None else len(surface_meshes)
        self.bc_coords = barycentric_explicit_to_implicit(bc_coords) if bc_coords.shape[1] > 2 else bc_coords
        self.bc_dirs = barycentric_explicit_to_implicit(bc_dirs) if bc_dirs.shape[1] > 2 else bc_dirs
        self.bc_fids = bc_fids
        self.lengths = lengths
        self.widths = widths
        self.pre_strains = pre_strains
        self.states = states if states is not None else np.ones(self.n_frames, dtype=np.float64)
        self.surface_meshes = surface_meshes
        self._vertices_ref = vertices_ref

        # GUI / Renderables
        self._color_strain_multiplier = 0.25
        self._color_mode = 0  #[0] Default  [1] Energy Density
        self._meshes = None

        # Material design variables
        self._material_d1 = 0.1
        self._material_d2 = 1.0
        # Young moduli of materials 1 and 2
        self._material_ym1 = 0.5
        self._material_ym2 = 25.7
        # Thicknesses of materials 1 and 2 in meters
        self._material_t1 = 0.00027
        self._material_t2 = 0.00035

        # Derived Quantities (after generating geometry from bcs/curves)
        self.vertices = vertices
        self.faces = faces
        self.curve_points = None
        self.curve_bcs = None
        self.curve_fids = None
        self.n_vertices = None

        # If vertices are specified - the clutch has already been generated (and possibly simulated)
        if vertices is None:
            self.forward_geometry()
        else:
            self.remake_renderables()

        # Energy quantities
        self.cst_model = self.get_cst_model()
        self.F = None
        self.FT = None
        self.CG = None
        self.E = None
        self.energy_densities = None
        self.energy = None
        self.forward_energy()

    def forward(self):
        self.forward_geometry()
        self.forward_energy()

    def forward_geometry(self):
        """Generates clutches geometry (splines/meshes) from barycentric coordinates"""

        if self.surface_meshes is None:
            print("Surface Mesh required for Clutch geometry creation")
            return

        self.curve_points = []
        self.curve_bcs = []
        self.curve_fids = []
        self.vertices = None
        self.faces = None
        self.n_vertices = None

        for i in range(self.n_clutches):
            curve_points, curve_bcs, curve_fids, vertices, faces = clutch_model(
                                   to_t(self.bc_coords[i]).to(dtype=C.f_precision),
                                   to_t(self.bc_dirs[i]).to(dtype=C.f_precision),
                                   self.bc_fids[i],
                                   self.lengths[i],
                                   to_t(self.surface_meshes.vertices).to(dtype=C.f_precision),
                                   to_t(self.surface_meshes.faces).to(dtype=C.i_precision),
                                   to_t(self.surface_meshes.face_normals).to(dtype=C.f_precision))

            self.curve_points.append(to_n(curve_points))
            self.curve_bcs.append(to_n(curve_bcs))
            self.curve_fids.append(to_n(curve_fids))
            self.vertices = to_n(vertices) if self.vertices is None else np.concatenate((self.vertices, to_n(vertices)), axis=1)
            self.faces = to_n(faces) if self.faces is None else np.concatenate((self.faces, to_n(faces) + self.n_vertices.sum() if self.n_vertices is not None else 0), axis=0)
            self.n_vertices = np.array([vertices.shape[1]]) if self.n_vertices is None else np.concatenate((self.n_vertices, np.array([vertices.shape[1]])))

        self._vertices_ref = self.vertices[0] if self.vertices is not None else None
        self.remake_renderables()
        self.cst_model = self.get_cst_model()

    def forward_energy(self):
        """Computes energy terms given rest/deformed geometries using the CST model"""
        
        if self.n_clutches > 0:
            # Get deformation gradient and principle components from model
            F, FT, CG, E, energy_densities, energy, compression = self.cst_model(
                torch.from_numpy(self.vertices).to(device=C.device, dtype=C.f_precision),
                youngmoduli=torch.from_numpy(self.youngmoduli).to(device=C.device, dtype=C.f_precision),
                thicknesses=torch.from_numpy(self.thicknesses).to(device=C.device, dtype=C.f_precision),
                return_quantities=True
                )

            self.F = to_n(F)
            self.FT = to_n(FT)
            self.CG = to_n(CG)
            self.E = to_n(E)
            self.energy_densities = to_n(energy_densities)
            self.energy = to_n(energy)

            self._meshes.face_colors = self.get_face_colors()
            self._meshes.redraw()

    def remake_renderables(self):
        self.nodes = []

        if self.n_clutches > 0:
            self._meshes = Meshes(self.vertices.copy(), faces=self.faces.copy(),  color=(0.0, 1.0, 0.5, 0.8), is_selectable=False)
            self._meshes.vertices += self._meshes.vertex_normals * 0.001
            self.add(self._meshes, show_in_hierarchy=False)

            if self.curve_points is not None:
                for i in range(self.n_clutches):
                    lines = Lines(self.curve_points[i], r_base=0.00025, color=(0.0, 0.0, 1.0, 1.0), is_selectable=False)
                    spheres = Spheres(self.endpoints[i], radius=0.001, color=(0.0, 1.0, 0.0, 1.0), is_selectable=False)
                    self.add(lines, spheres, show_in_hierarchy=False)

    @property
    def n_clutches(self):
        return len(self.bc_coords)

    @property
    def endpoints(self):
        if len(self.curve_points) == 0:
            return None
        # Lines x N_Frames x 2 x 3
        return np.stack([p[:, [0, -1]] for p in self.curve_points], axis=0)

    @property
    def n_vertices_total(self):
        return self.vertices.shape[1]

    @property
    def n_faces(self):
        return self.faces.shape[0]

    @property
    def vertices_ref(self):
        # Make sure ref vertices are set
        if self._vertices_ref is None:
            self._vertices_ref = self.vertices[0]

        if self.n_vertices is None:
            return self._vertices_ref

        pre_strain_per_vertex = np.ones(self.n_vertices_total)
        start=0
        for i in range(len(self.n_vertices)):
            end = start + self.n_vertices[i]
            pre_strain_per_vertex[start:end] *= self.pre_strains[i]
            start = start + self.n_vertices[i]
        return self._vertices_ref * pre_strain_per_vertex[...,np.newaxis]

    @property
    def youngmoduli(self):
        youngmoduli = np.zeros((self.n_frames, self.n_faces), dtype=np.float64)
        youngmoduli[self.states == self._material_d1] = self._material_ym1
        youngmoduli[self.states == self._material_d2] = self._material_ym2
        return youngmoduli

    @property
    def thicknesses(self):
        thicknesses = np.zeros((self.n_frames, self.n_faces), dtype=np.float64)
        thicknesses[self.states == self._material_d1] = self._material_t1
        thicknesses[self.states == self._material_d2] = self._material_t2
        return thicknesses

    @property
    def attach_vertex_indices(self):
        """Indices of vertices which correspond to attach points"""
        if self.vertices is None:
            return None
        avi = None
        for i in range(len(self.n_vertices)):
            v_offset = int(self.n_vertices[i]/3)
            vi = np.array([0, v_offset-1, v_offset, v_offset *2, v_offset * 2 - 1, v_offset * 3 - 1]).astype(np.int64) + self.n_vertices[:i].sum()
            avi = vi if avi is None else np.concatenate((avi, vi))
        return avi

    @property
    def attach_fids(self):
        """Face ids which correspond to attach points"""
        if self.vertices is None:
            return None
        return np.concatenate([cfid[[0, -5, -4, -3, -2, -1]] for cfid in self.curve_fids], axis=0)

    @property
    def attach_bcs(self):
        """Face ids which correspond to attach points"""
        if self.vertices is None:
            return None
        return np.concatenate([cbcs[[0, -5, -4, -3, -2, -1]] for cbcs in self.curve_bcs], axis=0)

    @property
    def attach_vertex_indices_center(self):
        """Indices of vertices which correspond to attach points"""
        if self.vertices is None:
            return None
        avic = None
        for i in range(len(self.n_vertices)):
            v_offset = int(self.n_vertices[i]/3)
            vic = np.arange(0, v_offset).astype(np.int64) + self.n_vertices[:i].sum()
            avic = vic if avic is None else np.concatenate((avic, vic))
        return avic

    @property
    def attach_vertex_indices_left(self):
        """Indices of vertices which correspond to attach points"""
        if self.vertices is None:
            return None
        avic = None
        for i in range(len(self.n_vertices)):
            v_offset = int(self.n_vertices[i]/3)
            vic = np.arange(v_offset, v_offset*2).astype(np.int64) + self.n_vertices[:i].sum()
            avic = vic if avic is None else np.concatenate((avic, vic))
        return avic


    @property
    def attach_vertex_indices_right(self):
        """Indices of vertices which correspond to attach points"""
        if self.vertices is None:
            return None
        avic = None
        for i in range(len(self.n_vertices)):
            v_offset = int(self.n_vertices[i]/3)
            vic = np.arange(v_offset*2, v_offset*3).astype(np.int64) + self.n_vertices[:i].sum()
            avic = vic if avic is None else np.concatenate((avic, vic))
        return avic


    @property
    def attach_fids_center(self):
        """Face ids which correspond to attach points"""
        if self.vertices is None:
            return None
        afc = None
        for i in range(len(self.n_vertices)):
            n_vertices = int(self.n_vertices[i]/3)
            fc = self.curve_fids[i][np.arange(0, n_vertices)]
            afc = fc if afc is None else np.concatenate((afc, fc))
        return afc

    @property
    def attach_bcs_center(self):
        """Face ids which correspond to attach points"""
        if self.vertices is None:
            return None
        abc = None
        for i in range(len(self.n_vertices)):
            n_vertices = int(self.n_vertices[i]/3)
            bc = self.curve_bcs[i][np.arange(0, n_vertices)]
            abc = bc if abc is None else np.concatenate((abc, bc))
        return abc

    @property
    def f_rest_areas(self):
        return to_n(self.cst_model.f_rest_areas)

    @property
    def area(self):
        return to_n(self.cst_model.f_rest_areas.sum())

    # @property
    # def area_current(self):
    #     return (self.f_rest_areas * self.densities[self.current_frame_id].astype(np.int64)).sum(-1)

    @property
    def area_current_percent(self):
        return self.area_current / self.area

    @property
    def color_mode(self):
        return self._color_mode

    @color_mode.setter
    def color_mode(self, color_mode):
        self._color_mode = color_mode
        face_colors = self.get_face_colors()
        self._meshes.face_colors = face_colors
        self._meshes.redraw()

    def get_cst_model(self):
        if self.n_clutches > 0:
            vertices_ref = torch.from_numpy(self.vertices_ref).to(device=C.device, dtype=C.f_precision)
            faces = torch.from_numpy(self.faces).to(device=C.device, dtype=C.i_precision)
            return CSTri(faces=faces, vertices_ref=vertices_ref)
        return None

    def get_face_colors(self):
        if self._color_mode == 1:
            # Strain coloring according to energy
            c = colors.Normalize(vmin=0.0, vmax=0.1 / self._color_strain_multiplier)(self.energy_densities)
            # Shift neutral to blue and invert direction
            f_colors = cm.hsv((2 / 3 - c))
        else:
            # default color
            f_colors = np.zeros_like(self.energy_densities)[...,np.newaxis].repeat(4,2)
            f_colors[:, :, 1] += 1.0
            f_colors[:, :, 2] += 0.5
            f_colors[:, :, 3] += 0.8

        return f_colors

    def nearest_clutch(self, query, min_dist=0.005):
        if self.n_clutches > 0:
            dists = np.linalg.norm(query - self.endpoints[:, self.current_frame_id, 0], axis=-1)
            if dists.min() < min_dist:
                return dists.argmin()
            return None

    def add_clutch(self, bc_coord, bc_fid):
        self.bc_coords = np.concatenate((self.bc_coords, barycentric_explicit_to_implicit(bc_coord[np.newaxis])))
        self.bc_fids = np.concatenate((self.bc_fids, np.array([bc_fid])))
        self.bc_dirs = np.concatenate((self.bc_dirs, np.array([0.5, 0.5])[np.newaxis]))
        self.lengths = np.concatenate((self.lengths, np.array([0.1])))
        self.pre_strains = np.concatenate((self.pre_strains, np.array([1.0])))

        self.forward()

    def remove_clutch(self, clutch_id):
        self.bc_coords = np.delete(self.bc_coords, clutch_id, axis=0)
        self.bc_fids = np.delete(self.bc_fids, clutch_id, axis=0)
        self.bc_dirs = np.delete(self.bc_dirs, clutch_id, axis=0)
        self.lengths = np.delete(self.lengths, clutch_id, axis=0)
        self.pre_strains = np.delete(self.pre_strains, clutch_id, axis=0)

        self.forward()

    def move_clutch(self, clutch_id, bc_fid, bc_coord):
        # clutch = self.clutches[clutch_id]

        # Convert bc_dir to eucledean world_dir
        curr_fid = self.bc_fids[clutch_id]
        curr_bc = self.bc_coords[clutch_id]
        curr_dir = self.bc_dirs[clutch_id]

        # Calculate start/mid/end points in euclidean space
        tri_curr_face = self.surface_meshes.vertices[[self.current_frame_id]][:, self.surface_meshes.faces[curr_fid]]
        p_start = barycentric_to_points(triangles=tri_curr_face, barycentric=curr_bc)
        p_mid = barycentric_to_points(triangles=tri_curr_face, barycentric=curr_dir)
        world_dir = p_mid - p_start
        world_dir = world_dir / np.linalg.norm(world_dir)
        target_bc_dist = np.linalg.norm(p_mid - p_start)

        # Move bc coords
        self.bc_coords[clutch_id] = bc_coord[:-1]
        self.bc_fids[clutch_id] = bc_fid

        # Convert back to bc space in (possibly) new triangle
        p_start_new = barycentric_to_points(triangles=tri_curr_face, barycentric=self.bc_coords[clutch_id])
        self.bc_dirs[clutch_id] = points_to_barycentric( triangles=tri_curr_face, points=p_start_new + world_dir * target_bc_dist)[0, :-1]

        self.forward()

    @property
    def energies(self):
        return self.energy.sum(-1)

    @property
    def energy_over_area(self):
        return self.energy.sum(-1) / self.area

    @property
    def geodesic_lengths_curve_points(self):
        return np.stack([np.linalg.norm(p[:, :-1] - p[:, 1:], axis=-1).sum(-1) for p in self.curve_points], axis=0)

    @property
    def curr_geodesic_lengths(self):
        return self.geodesic_lengths[self.current_frame_id]

    @property
    def geodesic_lengths(self):
        """Compute length of meshed """
        if self.vertices is None:
            return None

        vl = (self.n_vertices / 3).astype(np.int32)
        vs = self.vertices[:, self.attach_vertex_indices_center]

        min_index = 0
        max_index = vl[0]
        lengths = None
        for i in range(len(self.n_vertices)):
            length_c_i = np.linalg.norm(vs[:, min_index:max_index][:, :-1] - vs[:, min_index:max_index][:, 1:], axis=-1).sum(-1)[np.newaxis]
            lengths = length_c_i if lengths is None else np.concatenate((lengths, length_c_i), axis=0)
            if i < len(vl)-1:
                min_index += vl[i]
                max_index += vl[i+1]
        return lengths.transpose()

    def gui(self, imgui):
        super().gui(imgui)

        # Coloring options for all clutches
        csmu, color_strain_multiplier = imgui.slider_float('Color Strain ##c_multiplier{}'.format(self.unique_name),
                                                               self._color_strain_multiplier, 0.01, 1.0, '%.3f')
        cmu, color_mode = imgui.combo('Color Mode ##color_mode{}'.format(
            self.unique_name), self._color_mode, ['Default Color', 'Energy Density'])
        if cmu or csmu:
            self._color_strain_multiplier = color_strain_multiplier
            self.color_mode = color_mode

        # Per-clutch controls
        for i in range(self.n_clutches):
            imgui.text("Clutch {}".format(i+1))
            bc_dir_u, bc_dir = imgui.drag_float2('BC Dirs ##bc_dirs{}{}'.format(self.unique_name, i),
                                                               *self.bc_dirs[i], 0.01, -0.99, 0.99, '%.3f')
            length_u, length = imgui.drag_float('Length ##length{}{}'.format(self.unique_name, i),
                                                               self.lengths[i], 0.01, 0.0001, 1.0, '%.3f')
            if bc_dir_u or length_u:
                self.lengths[i] = length
                self.bc_dirs[i] = np.array(bc_dir)
                self.forward_geometry()

            pre_strain_u, pre_strain = imgui.drag_float('Pre-Strain ##prestrain{}{}'.format(self.unique_name, i),
                                                               self.pre_strains[i], 0.01, 0.01, 1.0, '%.3f')
            if pre_strain_u:
                self.pre_strains[i] = pre_strain

            imgui.plot_lines("Length ", self.geodesic_lengths[:, i].astype(np.float32))

    def gui_context_menu(self, imgui, *menu_position):
       pass

    def gui_io(self, imgui):
        if imgui.button("Export Sequence (.npz)"):
            self.to_npz()
        if imgui.button("Export Ref Frame (.npz"):
            self.to_npz(save_frames=False)

        if imgui.button("Export Lengths (.csv)"):
            self.save_lengths()

    def save_lengths(self, path=os.path.join(C.datasets.kg, "synthetic_data/clutch_lengths/")):
        np.savetxt(path + self.name + "_lengths.csv", self.geodesic_lengths, delimiter=",")

    def to_npz(self, path=os.path.join(C.datasets.kg, "clutches/"), save_frames=True):
        np.savez_compressed(path + self.name + '.npz',
                            bc_coords=self.bc_coords,
                            bc_fids=self.bc_fids,
                            bc_dirs=self.bc_dirs,
                            lengths=self.lengths,
                            pre_strains=self.pre_strains,
                            states=self.states,
                            vertices=self.vertices if save_frames else None,
                            faces=self.faces if save_frames else None,
                            vertices_ref=self.vertices_ref if save_frames else None
                            )

    @classmethod
    def from_npz(cls, npz_path, surface_mesh=None, name=None):
        """Load clutches from an npz file. The filename becomes the name of the sequence"""
        data = np.load(npz_path, allow_pickle=True)
        name = os.path.splitext(os.path.basename(npz_path))[0] if name is None else name

        bc_coords = data['bc_coords'] if 'bc_coords' in data and data['bc_coords'].dtype != object else None
        bc_fids = data['bc_fids'] if 'bc_fids' in data and data['bc_fids'].dtype != object else None
        bc_dirs = data['bc_dirs'] if 'bc_dirs' in data and data['bc_dirs'].dtype != object else None
        lengths = data['lengths'] if 'lengths' in data and data['lengths'].dtype != object else None
        pre_strains = data['pre_strains'] if 'pre_strains' in data and data['pre_strains'].dtype != object else np.ones(len(lengths))
        states = data['states'] if 'states' in data and data['states'].dtype != object else None
        vertices = data['vertices'] if 'vertices' in data and data['vertices'].size > 1 else None
        faces = data['faces'] if 'faces' in data and data['faces'].dtype != object else None
        vertices_ref = data['vertices_ref'] if 'vertices_ref' in data and data['vertices_ref'].size > 1 else None

        return cls(
            bc_coords=bc_coords,
            bc_fids=bc_fids,
            bc_dirs=bc_dirs,
            lengths=lengths,
            pre_strains=pre_strains,
            states=states if surface_mesh is None else None,
            surface_meshes=surface_mesh,
            vertices=vertices,
            faces=faces,
            vertices_ref=vertices_ref,
            name=name
        )

