import os
import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.colors as colors
from aitviewer.renderables.meshes import Meshes
from cdkg.configuration import CONFIG as C
import trimesh
import meshio

class TetMeshes(Meshes):
    """A sequence of tetrahedral meshes. This assumes that the tet mesh topology is fixed over the sequence."""

    def __init__(self,
                 vertices,
                 elements,
                 outer_faces=None,
                 element_colors=None,
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

        # Create faces from tetrahedral elements
        assert len(elements.shape) == 2
        faces = np.concatenate((
            elements[:, [2, 1, 0]],
            elements[:, [1, 2, 3]],
            elements[:, [0, 3, 2]],
            elements[:, [3, 0, 1]]
        ))

        super(TetMeshes, self).__init__(vertices=vertices, faces=faces, *args, **kwargs)
        self.elements = elements
        self.element_colors = element_colors
        self.outer_faces = outer_faces


    @property
    def element_colors(self):
        return self._element_colors

    @element_colors.setter
    def element_colors(self, element_colors):
        self._element_colors = element_colors
        if element_colors is not None:
            element_colors = element_colors[np.newaxis] if element_colors.ndim == 2 else element_colors
            self.face_colors = np.tile(element_colors, (1,4,1))

    def to_npz(self, path=os.path.join(C.smpl_tet, "npz/")):
        np.savez_compressed(path + self.name + '.npz',
                            vertices=self.vertices,
                            faces=self.faces,
                            )

    def to_vtk(self, path=os.path.join(C.smpl_tet, "vtk/")):
        mesh = meshio.Mesh(points=self.current_vertices, cells=[("tetra", self.elements)])
        mesh.write(path + self.name + '.vtk')

    @classmethod
    def from_vtk(cls, path, frames=1):
        mesh = meshio.read(path)
        v = mesh.points.astype(np.float32)
        if frames > 1:
            if len(v.shape) == 2:
                v = v[np.newaxis]
            v = np.repeat(v, frames, axis=0)
        return cls(vertices=v, elements=mesh.cells_dict['tetra'].astype(np.int64))

    @classmethod
    def from_npz(cls, path):
        pass

    def gui_io(self, imgui):
        if imgui.button("Export Current Frame(.vtk)"):
            self.to_vtk()
        if imgui.button("Export Sequence (.npz)"):
            self.to_npz()
