import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.colors as colors
from cdkg.renderables.tet_meshes import TetMeshes
from cdkg.utils import to_n, to_t
from cdkg.models.cs_tet import CSTet
from aitviewer.configuration import CONFIG as C

class SoftBody(TetMeshes):
    """A sequence of tetrahedral meshes. This assumes that the tet mesh topology is fixed over the sequence."""

    def __init__(self,
                 youngs_modulus=5.7,
                 *args,
                 **kwargs
                 ):
        """
        :param vertices: A np array of shape (N, V, 3) or (V, 3).
        :param elements: A np array of shape (E, 4) denoting the 4 vertices of the tetrahedrons
        :param element_colors: A np array of shape (N, E, 4).
        :param args: Pass onto parent (meshes)
        :param kwargs: Pass onto parent (meshes)
        :param vertices_ref: The rest state of the garment
        """

        super(SoftBody, self).__init__(*args, **kwargs)

        # GUI Only
        self._color_strain_multiplier = 0.25
        self._color_mode = 0  # [0] Energy Density Psi, [1] Frobeneus Norm

        # Material design variables
        self.youngs_modulus = youngs_modulus

        # Energy Quantities
        self.cstet_model = self.get_model()
        self.CG = None
        self.E = None
        self.energy_densities = None
        self.energy = None

        self.forward_energy()

    def forward_energy(self):
        # Get deformation gradient and principle components from model
        CG, E, energy_densities, energy = self.cstet_model(
            torch.from_numpy(self.vertices).to(device=C.device, dtype=C.f_precision),
            youngmoduli=torch.from_numpy(self.youngmoduli).to(device=C.device, dtype=C.f_precision),
            return_quantities=True
        )
        self.CG = to_n(CG)
        self.E = to_n(E)
        self.energy_densities = to_n(energy_densities)
        self.energy = to_n(energy)
        self.update_strain_color()

    @property
    def youngmoduli(self):
        youngmoduli = np.zeros(self.elements.shape[0], dtype=np.float64)
        youngmoduli[:] = self.youngs_modulus
        return youngmoduli

    @property
    def f_rest_volumes(self):
        return to_n(self.cstet_model.f_rest_areas)

    @property
    def volume(self):
        return to_n(self.cstet_model.f_rest_areas.sum())

    @property
    def vertices_ref(self):
        return self.vertices[0]

    @property
    def color_mode(self):
        return self._color_mode

    @color_mode.setter
    def color_mode(self, color_mode):
        self._color_mode = color_mode
        self.element_colors = self.get_element_colors()
        self.redraw()

    def get_model(self):
        vertices_ref = torch.from_numpy(self.vertices_ref).to(device=C.device, dtype=C.f_precision)
        elements = torch.from_numpy(self.elements).to(device=C.device, dtype=C.i_precision)
        return CSTet(elements=elements, vertices_ref=vertices_ref)

    def get_element_colors(self):
        if self._color_mode == 1:
            # Frobeneus Norm colors
            frob_norm_sq = np.power(self.CG - np.eye(3), 2).sum(-1).sum(-1)
            frob_norm_sq = colors.Normalize(vmin=0, vmax=1.0)(frob_norm_sq)
            e_colors = cm.hsv((2 / 3 - frob_norm_sq))
        else:
            # Energy Density
            c = colors.Normalize(vmin=0.0, vmax=0.1 / self._color_strain_multiplier )(self.energy_densities)
            # Shift neutral to blue and invert direction
            e_colors = cm.hsv((2 / 3 - c))
        return e_colors

    def gui(self, imgui):
        super().gui(imgui)

        # Trigger face color update
        csmu, color_strain_multiplier = imgui.slider_float('Color Strain ##c_multiplier{}'.format(self.unique_name),
                                                               self._color_strain_multiplier, 0.01, 1.0, '%.3f')
        cmu, color_mode = imgui.combo('Color Mode ##color_mode{}'.format(
            self.unique_name), self._color_mode, ['Energy Density',
                                                  'Frobeneus Norm'
                                                  ])

        if cmu or csmu:
            self._color_strain_multiplier = color_strain_multiplier
            self.color_mode = color_mode

        # imgui.text("Current Energy {:.7f}".format(self.dense_energies[self.current_frame_id] * 1e6))
        # imgui.text("Current Volume {:.3f}".format(self.volume))


    def update_strain_color(self):
        self.element_colors = self.get_element_colors()
        self.redraw()

    # def make_renderable(self, ctx):
    #     # Force redraw after setting up buffers
    #     super().make_renderable(ctx)
    #     self.update_strain_color()



