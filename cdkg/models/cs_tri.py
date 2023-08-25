import torch
from cdkg.configuration import CONFIG as C

class CSTri(torch.nn.Module):
    """
    Computes deformation gradient F and related quantities on input triangular mesh
    """

    def __init__(self, faces, vertices_ref, material="neo_hookean"):
        """
        :param vertices_ref: (1, N, V, 3) Reference vertices of the mesh (single set)
        :param faces: (N, F, 3) Faces of the mesh (topology does not change)
        :param material: Constitutive model energy. Currently implemented: neo_hookean, stvk. Default: neo_hookean
        """

        super(CSTri, self).__init__()

        self.faces = faces
        self.vertices_ref = vertices_ref
        self.edges_ref_2d = CSTri.get_edges_2d(vertices_ref, faces)
        self.edges_ref_2d_inv = torch.inverse(self.edges_ref_2d)
        self.material = material
        self.f_rest_areas = torch.abs(torch.linalg.det(self.edges_ref_2d))*0.5

    def forward(self, vertices, youngmoduli=None, thicknesses=None, return_quantities=False, relax=True):
        """
        :param vertices: (B, N, V, 3) Deformed vertices of the mesh (batched set)
        :return: Total energy of the constitutive model
        """
        F, FT = self.get_deformation_gradient(vertices)
        CG, E = self.get_strain_measures(F, FT)

        energy_densities, compression = self.apply_material(CG, E, youngmoduli=youngmoduli, relax=relax)
        energies = energy_densities * self.f_rest_areas * thicknesses

        if return_quantities:
            return F, FT, CG, E, energy_densities, energies, compression

        return energies.sum(-1)


    def apply_material(self, CG, E, youngmoduli, relax=True):
        if self.material == "neo_hookean":
            return self._neo_hookean(CG, youngmoduli, relax=relax)
        elif self.material == "stvk":
            return self._StVK_material(E, youngmoduli, relax=relax)
        raise ValueError('No constitutive model found for specified material.')

    def get_deformation_gradient(self, vertices):
        """Get the deformation gradient"""
        edges_def_all = vertices[:, self.faces]
        edges_def = (edges_def_all - edges_def_all[:, :, [0]])[:, :, 1:]

        # F = torch.matmul(self.edges_ref_2d_inv.unsqueeze(0), edges_def)
        # FT = torch.transpose(F, 2, 3)

        # Alternative way to compute which makes more sense
        FT = torch.matmul(edges_def.transpose(2,3), self.edges_ref_2d_inv.transpose(1,2).unsqueeze(0))
        F = torch.transpose(FT, 2, 3)

        # Check that it works: this should be ~0
        # torch.matmul(FT, self.edges_ref_2d) - edges_def.transpose(1, 2)

        return F, FT

    def get_strain_measures(self, F, FT):
        # Cauchy-green deformation tensor
        # CG = torch.matmul(F, FT)

        # Computing F*FT is much faster if we take a bunch of dot products and squares instead of matrix multiply
        CG = torch.zeros((FT.shape[0], FT.shape[1], 2, 2), device=C.device, dtype=C.f_precision)
        off_d = (FT[:, :, :, 0] * FT[:, :, :, 1]).sum(-1)
        CG[:, :, 0, 0] = (FT[:, :, :, 0]**2).sum(-1)
        CG[:, :, 0, 1] = off_d
        CG[:, :, 1, 0] = off_d
        CG[:, :, 1, 1] = (FT[:, :, :, 1]**2).sum(-1)

        # Green-lagrange strain
        E = 0.5*(CG - torch.eye(2,2, device=C.device, dtype=C.f_precision).unsqueeze(0))

        return CG, E

    def get_principle_components(self, strain_tensor, eigvecs=False):
        # Main stretch components of the Deformation Tensor

        if eigvecs:
            # run eigh on the cpu only when we need to. too slow on the gpu
            eig_vals, eig_vecs = torch.linalg.eigh(strain_tensor.to(device="cpu"))
            eig_vals, eig_vecs = eig_vals.to(device=C.device), eig_vecs.to(device=C.device)
            return eig_vals, eig_vecs
        else:
            # Compute eigenvalues analytically for 2x2 matrix:
            # https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
            tr = torch.diagonal(strain_tensor, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
            d = torch.det(strain_tensor).unsqueeze(-1)
            a = torch.pow(tr,2)/4-d
            a[a<1e-15] = 1e-15 # clip negative and too small numbers as they are problematic for sqrt
            rh = torch.pow(a, 0.5)
            eig1 = tr/2 - rh
            eig2 = tr/2 + rh
            # Do not want negative values
            eig1[eig1<1e-15] = 1e-15 # clip negative and too small numbers as they are problematic for sqrt
            eig2[eig2<1e-15] = 1e-15 # clip negative and too small numbers as they are problematic for sqrt

            return torch.cat((eig1, eig2), dim=-1)

    @staticmethod
    def get_edges_2d(vertices, faces):
        edges_ref = (vertices[faces] - vertices[faces][:, [0]])[:, 1:]
        return CSTri.edges_3d_to_2d(edges_ref)

    @staticmethod
    def edges_3d_to_2d(edges):
        """
        :param edges: Edges in 3D space (in the world coordinate basis) (E, 2, 3)
        :return: Edges in 2D space (in the intrinsic orthonormal basis) (E, 2, 2)
        """
        # Decompose for readability
        edges0 = edges[:, 0]
        edges1 = edges[:, 1]

        # Get orthonormal basis
        basis2d_0 = (edges0 / torch.norm(edges0, dim=-1).unsqueeze(-1))
        n = torch.cross(basis2d_0, edges1, dim=-1)
        basis2d_1 = torch.cross(n, edges0, dim=-1)
        basis2d_1 = basis2d_1 / torch.norm(basis2d_1, dim=-1).unsqueeze(-1)

        # Project original edges into orthonormal basis
        edges2d = torch.zeros((edges.shape[0], edges.shape[1], 2)).to(device=C.device, dtype=C.f_precision)
        edges2d[:, 0, 0] = (edges0 * basis2d_0).sum(-1)
        edges2d[:, 0, 1] = (edges0 * basis2d_1).sum(-1)
        edges2d[:, 1, 0] = (edges1 * basis2d_0).sum(-1)
        edges2d[:, 1, 1] = (edges1 * basis2d_1).sum(-1)

        return edges2d

    def _convert_youngsmodulus_poissonsratio_to_lame_params(self, ym, pr):
        lam = ym * pr / ((1+pr) * (1-2 * pr))
        mu = ym / (2 * (1 + pr))
        return lam, mu

    def _neo_hookean(self, CG, youngmoduli=0.05, poisson_ratio=0.33, relax=False):
        """YM is in MPa mega-pascals"""

        lam, mu = self._convert_youngsmodulus_poissonsratio_to_lame_params(youngmoduli, poisson_ratio)

        if relax:
            evals = self.get_principle_components(CG)
            evals_max = evals[:, :, -1].clone()
            evals_min = evals[:, :, 0].clone()
            evals_min_tilda = 1 / torch.sqrt(evals_max)

            # both evals < 0
            c0_mask = evals_max < 1

            # evals_max >= 1 and evals_min_tilda > evals_min
            c1_mask = torch.logical_and(evals_max >= 1, evals_min_tilda > evals_min)
            evals_min[c1_mask] = evals_min_tilda[c1_mask]

            ic = evals_max + evals_min
            iic = evals_max * evals_min
            j = torch.sqrt(iic)
            log_j = torch.log(j)
            energy_density = 0.5*mu*(ic - 2.0) - mu*log_j + 0.5*lam*torch.pow(log_j, 2)
            energy_density[c0_mask] = 0.0

            # compression = evals_min.clone().detach()
            # compression[~c1_mask] = 1.0
            # compression[c0_mask] = 1.0

            return energy_density, evals_min


        ic = torch.diagonal(CG, dim1=-2, dim2=-1).sum(-1)  # Trace
        iic = torch.linalg.det(CG)
        j = torch.sqrt(iic)
        log_j = torch.log(j)
        return 0.5*mu*(ic - 2.0) - mu*log_j + 0.5*lam*torch.pow(log_j, 2), None

    def _StVK_material(self, E, ym=0.05, pr=0.33, relax=False):
        """YM is in MPa mega-pascals"""
        lam, mu = self._convert_youngsmodulus_poissonsratio_to_lame_params(ym, pr)
        trace_E = torch.diagonal(E, dim1=-2, dim2=-1).sum(-1)  # Trace
        volume_change = torch.pow(trace_E, 2)
        all_change = torch.diagonal(torch.pow(E, 2), dim1=-2, dim2=-1).sum(-1)  # torch.pow(E, 2).sum(-1).sum(-1)
        return 0.5*lam*volume_change + mu*all_change, None

    def __len__(self):
        return self.vertices.shape[0]
