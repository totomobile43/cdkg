import torch
from cdkg.configuration import CONFIG as C

class CSTet(torch.nn.Module):
    """
    Computes deformation gradient F and related quantities on input tetrehedral mesh
    """

    def __init__(self, elements, vertices_ref, material="neo_hookean"):
        """
        :param vertices_ref: (1, N, V, 3) Reference vertices of the mesh (single set)
        :param elements: (N, T, 4) Tetrahedrons of the mesh (topology does not change)
        :param material: Constitutive model energy. Currently implemented: neo_hookean, stvk. Default: neo_hookean
        """

        super(CSTet, self).__init__()

        self.elements = elements
        self.vertices_ref = vertices_ref
        self.edges_ref = CSTet.get_edges(vertices_ref, elements)
        self.edges_ref_inv = torch.inverse(self.edges_ref)
        self.material = material
        self.f_rest_volumes = torch.abs(torch.linalg.det(self.edges_ref)/6.0)

    def forward(self, vertices, youngmoduli=None, return_quantities=False):
        """
        :param vertices: (B, N, V, 3) Deformed vertices of the tet_mesh (batched set)
        :return: Total energy of the constitutive model for all batches
        """
        F, FT = self.get_deformation_gradient(vertices)
        CG, E = self.get_strain_measures(F, FT)

        energy_densities = self.apply_material(CG, E, youngmoduli=youngmoduli)
        energies = energy_densities * self.f_rest_volumes

        if return_quantities:
            return CG, E, energy_densities, energies

        return energies.sum(-1)


    def apply_material(self, CG, E, youngmoduli):
        if self.material == "neo_hookean":
            return self._neo_hookean(CG, youngmoduli)
        elif self.material == "stvk":
            return self._StVK_material(E, youngmoduli)
        raise ValueError('No constitutive model found for specified material.')

    def get_deformation_gradient(self, vertices):
        """Get the deformation gradient"""
        elem_v = vertices[:, self.elements]
        edges_def = (elem_v - elem_v[:, :, [0]])[:, :, 1:]

        F = torch.matmul(edges_def, self.edges_ref_inv.unsqueeze(0))
        FT = torch.transpose(F, 2, 3)

        # Alternative way to compute which makes more sense
        # FT = torch.matmul(edges_def.transpose(2,3), self.edges_ref_2d_inv.transpose(1,2).unsqueeze(0))
        # F = torch.transpose(FT, 2, 3)

        # Check that it works: this should be ~0
        # (torch.matmul(F, self.edges_ref)  - edges_def).max()

        return F, FT

    def get_strain_measures(self, F, FT):
        # Cauchy-green deformation tensor
        CG = torch.matmul(F, FT)

        # Green-lagrange strain
        E = 0.5*(CG - torch.eye(3,3, device=C.device, dtype=C.f_precision).unsqueeze(0))

        return CG, E

    @staticmethod
    def get_edges(vertices, elements):
        return (vertices[elements] - vertices[elements][:, [0]])[:, 1:]

    def _convert_youngsmodulus_poissonsratio_to_lame_params(self, ym, pr):
        lam = ym * pr / ((1+pr) * (1-2 * pr))
        mu = ym / (2 * (1 + pr))
        return lam, mu

    def _neo_hookean(self, CG, youngmoduli=0.05, poisson_ratio=0.33):
        """YM is in MPa mega-pascals"""

        lam, mu = self._convert_youngsmodulus_poissonsratio_to_lame_params(youngmoduli, poisson_ratio)
        ic = torch.diagonal(CG, dim1=-2, dim2=-1).sum(-1)  # Trace
        iic = torch.linalg.det(CG)
        j = torch.sqrt(iic)
        log_j = torch.log(j)
        return 0.5*mu*(ic - 3.0) - mu*log_j + 0.5*lam*torch.pow(log_j, 2)

    def _StVK_material(self, E, ym=0.05, pr=0.33):
        """YM is in MPa mega-pascals"""
        lam, mu = self._convert_youngsmodulus_poissonsratio_to_lame_params(ym, pr)
        trace_E = torch.diagonal(E, dim1=-2, dim2=-1).sum(-1)  # Trace
        volume_change = torch.pow(trace_E, 2)
        all_change = torch.diagonal(torch.pow(E, 2), dim1=-2, dim2=-1).sum(-1)  # torch.pow(E, 2).sum(-1).sum(-1)
        return 0.5*lam*volume_change + mu*all_change

    def __len__(self):
        return self.vertices.shape[0]
