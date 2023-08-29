import torch
from pytorch3d.ops import knn_points
from cdkg.configuration import CONFIG as C


class SDF(torch.nn.Module):
    """
    Given a mesh and a point cloud, apply a loss on the points which are inside the mesh.
    Loss is computed via an SDF (Signed Distance Field)
    """

    def __init__(self, vertices, vertex_normals, knn_K=60, cache_knn=True):
        super(SDF, self).__init__()

        # Body
        self.vertices = vertices.to(device=C.device, dtype=C.f_precision)
        self.vertex_normals = vertex_normals.to(device=C.device, dtype=C.f_precision)
        self.support_radii = (knn_points(self.vertices.to(torch.float32), self.vertices.to(torch.float32), K=7).dists[..., 1:]).mean(-1) * 2.0
        self.knn_K = knn_K

        # Cache
        self.cache_knn = cache_knn
        self.knn_idx = None
        self.v = None
        self.vn = None
        self.sr = None

    def forward(self, points, inside_only=False, lengths=None):
        '''
        Produces a signed distance field defined by a surface vertices + normals from a set of query of points
        Implementation based on the IMLS (Implicit Moving Least Squares) by Öztireli1 et al.
        "Feature Preserving Point Set Surfaces based on Non‐Linear Kernel Regression"

        Local support radius is set to 2 * the distance to the nearest vertex (set per vertex)

        :param points: (N,P,3) Points to apply SDF to
        :param vertices: (N, V, 3) Vertices of the surface
        :param vertex_normals: (N, V, 3) Vertex normals of the surface
        :param K: Number of nearest K points to base the implicit function on
        :param lengths: Lengths of the query points (only if each batch dimension has different lengths)
        :return: Signed distance values for each query point to the surface
        '''

        # return torch.pow(points, 2).sum()

        # Add batch dimension if missing
        points = points.unsqueeze(0) if len(points.shape) == 2 else points

        # Get SDF
        sdf = self.get_sdf(points)

        # Apply lengths
        if lengths is not None:
            qp_mask = (torch.arange(points.shape[1], device=points.device)[None] >= lengths[:, None])
            sdf[qp_mask] = 0.0

        # Positive elements (outside) the SDF have no energy.
        if inside_only:
            sdf[sdf > 0] = 0.0

        return torch.pow(sdf, 2).sum(-1)

    def setup_cache(self, points, vertices_ref=None):
        """Build a point-to-surface vertex cache."""
        if len(points.shape) < 3:
            if vertices_ref is None:
                vertices_ref = self.vertices[[0]]
            self.knn_idx = knn_points(points.unsqueeze(0), vertices_ref, K=self.knn_K).idx
            self.knn_idx = self.knn_idx.unsqueeze(0).tile((len(self), 1, 1))
        else:
            self.knn_idx = knn_points(points.to(torch.float32), self.vertices.to(torch.float32), K=self.knn_K).idx

        # Indices are a multidimensional mask. To apply mask: Flatten -> Select -> Unflatten
        indices = self.knn_idx.view(len(self), -1) + torch.arange(0, self.vertices.shape[1] * len(self),
                                                             self.vertices.shape[1],
                                                             device=self.vertices.device).unsqueeze(-1)
        self.v = self.vertices.reshape(-1, 3)[indices].view(len(self), -1, self.knn_K, 3)  # (F, P, K, 3)
        self.vn = self.vertex_normals.reshape(-1, 3)[indices].view(len(self), -1, self.knn_K, 3)  # (F, P, K, 3)
        self.sr = self.support_radii.reshape(-1, 1)[indices].view(len(self), -1, self.knn_K)

    def get_sdf(self, points):
        """ Get the signed distances from each point to the surface defined by the mesh """

        # Setup Cache
        if self.knn_idx is None:
            self.setup_cache(points)

        # Distance from every point to all k-nearest vertices
        p_v = points.unsqueeze(-2) - self.v  # (F, P, V, 3)
        dists = torch.pow(p_v, 2).sum(-1)

        inside_h_mask = (dists < self.sr)

        phi_v = ((1.0 - dists / self.sr)**2)**2  # (F, P, V)  # Faster than ^4

        # Support radius vanishes  (set to 0) beyond h. Setting to 0 causes nan's in backwards pass.
        phi_v = phi_v.masked_fill(~inside_h_mask, torch.tensor(1e-18))

        # Fastest batch dot product - see https://github.com/pytorch/pytorch/issues/18027
        # v_to_p_dot_n = torch.einsum('fbij,fbij->fbi', vn, p_v)
        v_to_p_dot_n = (self.vn * p_v).sum(-1)

        phi = (phi_v * v_to_p_dot_n).sum(dim=-1) / phi_v.sum(dim=-1)

        return phi

    def __len__(self):
        return self.vertices.shape[0]


