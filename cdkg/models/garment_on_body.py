import torch
from cdkg.models.sdf import SDF
from cdkg.models.cs_tri import CSTri
from cdkg.configuration import CONFIG as C
torch.manual_seed(0)
import math
from cdkg.utils import to_t, to_n, subdivide_meshes_torch, barycentric_to_points

class GarmentOnBody(torch.nn.Module):
    """
    Models a thin layer of flexible skin as it undergoes deformation on an implicit body surface
    """

    def __init__(self,
                 lbfgs_iter=100,
                 outer_iter=20,
                 w_strain=1.0,
                 w_attach=0.003,
                 w_sdf=0.005,
                 w_c_strain=1.0,
                 w_c_attach=1.0,
                 w_c_all_attach=0.001,
                 w_c_sdf=0.005,
                 batch_size=80,
                 sdf_inside_only=True,
        ):
        super(GarmentOnBody, self).__init__()

        # Weights - set relative to strain (1.0)
        self.w_strain = w_strain
        self.w_attach = w_attach
        self.w_sdf = w_sdf
        self.sdf_inside_only = sdf_inside_only

        self.w_c_strain = w_c_strain
        self.w_c_attach = w_c_attach
        self.w_c_all_attach = w_c_all_attach
        self.w_c_sdf = w_c_sdf

        # Loop / batch sizes
        self.lbfgs_iter = lbfgs_iter
        self.outer_iter = outer_iter
        self.batch_size = batch_size

    def forward(self, garment, body, outer_iter=None, translate_with_body=False):

        if outer_iter is not None:
            self.outer_iter = outer_iter

        mode = 'simulate' if garment.clutches is None else 'simulate_w_clutches'

        # Init parameters
        body_v = to_t(body.vertices).to(dtype=C.f_precision)
        body_v_ref = body_v[[0]]
        body_vn = to_t(body.vertex_normals).to(dtype=C.f_precision)
        body_f = to_t(body.faces).to(dtype=C.i_precision)
        num_frames = body_v.shape[0]
        garment_v = to_t(garment.vertices)[:num_frames].to(dtype=C.f_precision)
        garment_f = to_t(garment.faces).to(dtype=C.i_precision)
        garment_yms = to_t(garment.youngmoduli).to(dtype=C.f_precision)
        garment_ts = to_t(garment.thicknesses).to(dtype=C.f_precision)
        garment_att = to_t(garment.att).to(dtype=torch.bool)
        garment_ref = to_t(garment.vertices_ref).to(dtype=C.f_precision)
        garment_v_init = garment_v.clone()

        # Attachment points are on the body
        if garment.subdivide:
            body_v_sub, body_f_sub = subdivide_meshes_torch(body_v, body_f, garment.subdivide)
            v_mask = torch.unique(body_f_sub[garment.f_mask].reshape(-1))
            garment_v_init = body_v_sub[:, v_mask]

        if mode == 'simulate':
            # Batch
            for i in range(int(math.ceil(num_frames / self.batch_size))):
                s = int(i * self.batch_size)
                e = int(min(num_frames, (i + 1) * self.batch_size))
                # print('Simulating Frames {:3d} to {:3d}'.format(s, e))

                garment_v[s:e] = self.simulate(
                    garment_v[s:e],
                    garment_f,
                    garment_ref,
                    garment_yms[s:e],
                    garment_ts[s:e],
                    garment_att,
                    garment_v_init[s:e],
                    body_v[s:e],
                    body_vn[s:e],
                    body_v_ref
                )

                # Translate current vertices to next frame
                if i + 1 < num_frames and translate_with_body:
                    assert (self.batch_size == 1), "The option translate_with_body requires a batch size of 1."

                    # Translate garment vertices according to body motion
                    garment_v[s + 1:e + 1] = garment_v[s:e] + body_v_sub[s + 1:e + 1, v_mask] - body_v_sub[s:e, v_mask]

                    if i == 1:
                        self.outer_iter = 1

            garment.vertices = to_n(garment_v)
            garment.forward_energy()

        if mode == 'simulate_w_clutches':
            # Extract all clutch params and merge
            c = garment.clutches

            clutches_v = to_t(c.vertices)[:num_frames].to(dtype=C.f_precision)
            clutches_v_ref = to_t(c.vertices_ref).to(dtype=C.f_precision)
            clutches_f = to_t(c.faces).to(dtype=C.i_precision)
            clutches_yms = to_t(c.youngmoduli).to(dtype=C.f_precision)
            clutches_ts = to_t(c.thicknesses).to(dtype=C.f_precision)
            clutches_attach_v = to_t(c.attach_vertex_indices).to(dtype=C.i_precision)
            clutches_attach_fids = to_t(c.attach_fids).to(dtype=C.i_precision)
            clutches_attach_bcs = to_t(c.attach_bcs).to(dtype=C.f_precision)
            clutches_attach_v_center = to_t(c.attach_vertex_indices_center).to(dtype=C.i_precision)
            clutches_attach_fids_center = to_t(c.attach_fids_center).to(dtype=C.i_precision)
            clutches_attach_bcs_center = to_t(c.attach_bcs_center).to(dtype=C.f_precision)

            clutches_attach_v_left = to_t(c.attach_vertex_indices_left).to(dtype=C.i_precision)
            clutches_attach_v_right = to_t(c.attach_vertex_indices_right).to(dtype=C.i_precision)

            # Batch
            for i in range(int(math.ceil(num_frames / self.batch_size))):
                s = int(i * self.batch_size)
                e = int(min(num_frames, (i + 1) * self.batch_size))

                garment_v[s:e], clutches_v[s:e] = self.simulate_w_clutch(
                    garment_v[s:e],
                    garment_f,
                    garment_ref,
                    garment_yms[s:e],
                    garment_ts[s:e],
                    garment_att,
                    garment_v_init[s:e],
                    clutches_v[s:e],
                    clutches_v_ref,
                    clutches_f,
                    clutches_yms[s:e],
                    clutches_ts[s:e],
                    clutches_attach_v,
                    clutches_attach_fids,
                    clutches_attach_bcs,
                    clutches_attach_v_center,
                    clutches_attach_fids_center,
                    clutches_attach_bcs_center,
                    body_v[s:e],
                    body_vn[s:e],
                    body_v_ref
                )

                # Translate current vertices to next frame
                if i+1 < num_frames and translate_with_body:
                    assert(self.batch_size==1), "The option translate_with_body requires a batch size of 1."

                    # Translate garment vertices according to body motion
                    garment_v[s+1:e+1] = garment_v[s:e] + body_v_sub[s+1:e+1, v_mask]-body_v_sub[s:e, v_mask]

                    # Translate clutch vertices according to garment motion
                    triangles = garment_v[s:e+1,  garment_f[clutches_attach_fids_center]]
                    garment_points = barycentric_to_points(triangles=triangles.view((-1, 3, 3)), bc_coords=clutches_attach_bcs_center.repeat(2, 1)).view(2, -1, 3)
                    clutches_v[s+1:e+1, clutches_attach_v_center] = clutches_v[s:e, clutches_attach_v_center] + garment_points[[1]]- garment_points[[0]]
                    clutches_v[s+1:e+1, clutches_attach_v_left] = clutches_v[s:e, clutches_attach_v_left] + garment_points[[1]]- garment_points[[0]]
                    clutches_v[s+1:e+1, clutches_attach_v_right] = clutches_v[s:e, clutches_attach_v_right] + garment_points[[1]]- garment_points[[0]]

                    if i == 1:
                        self.outer_iter = 1
                        self.lbfgs_iter = 75

            garment.vertices = to_n(garment_v)
            garment.forward_energy()
            c.vertices = to_n(clutches_v)
            c.remake_renderables()
            c.forward_energy()


        if mode == 'forces':
            # Used to compute forces after simulation (dE/dX)
            cst_forces = torch.zeros_like(garment_v).to(dtype=C.f_precision, device=C.DEVICE)
            attach_forces = torch.zeros_like(garment_v).to(dtype=C.f_precision, device=C.DEVICE)
            sdf_forces = torch.zeros_like(garment_v).to(dtype=C.f_precision, device=C.DEVICE)
            # Batch
            for i in range(int(math.ceil(num_frames / self.batch_size))):
                s = int(i * self.batch_size)
                e = int(min(num_frames, (i + 1) * self.batch_size))
                cst_forces[s:e], attach_forces[s:e], sdf_forces[s:e] = self.forces(
                    garment_v[s:e],
                    garment_f,
                    garment_ref,
                    garment_yms[s:e],
                    garment_ts[s:e],
                    garment_att,
                    garment_v_init[s:e],
                    body_v[s:e],
                    body_vn[s:e],
                )

            return to_n(cst_forces), to_n(attach_forces), to_n(sdf_forces)


    def simulate(self, garment_v, garment_f, garment_ref, garment_yms, garment_ts, garment_att, garment_v_init, body_v, body_vn, body_v_ref):
        """
        Simulates a garment as it slides over the body
        """

        # Init Energies
        sdf = SDF(vertices=body_v, vertex_normals=body_vn)
        cst = CSTri(faces=garment_f, vertices_ref=garment_ref)

        # Optimization Params
        garment_v = garment_v.requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [garment_v],
            lr=1,
            tolerance_grad=1e-32,
            tolerance_change=1e-32,
            max_iter=self.lbfgs_iter,
            line_search_fn='strong_wolfe')

        def get_obj_vals():
            e_strain = cst(garment_v, youngmoduli=garment_yms, thicknesses=garment_ts)
            e_attach = self.attach(garment_att, garment_f, garment_v, garment_v_init) if torch.any(garment_att) else 0
            e_sdf = sdf(garment_v, inside_only=self.sdf_inside_only)
            obj_vals = e_strain * self.w_strain + e_attach * self.w_attach + self.w_sdf * e_sdf
            return obj_vals

        def get_obj_val():
            optimizer.zero_grad()
            obj_val = get_obj_vals().sum()
            obj_val.backward()

            # Grad Norm (check for convergence on garment/clutch vertices)
            # print('i-{:3d}. Sim Grad Norm LBFGS Eval: {:.16}'.format(i, torch.norm(garment_v.grad).cpu().numpy()))

            return obj_val

        for i in range(self.outer_iter):

            optimizer.step(get_obj_val)
            print('i-{:3d}. Sim Grad Norm: {:.16}'.format(i, torch.norm(garment_v.grad).cpu().numpy()))

        return garment_v.detach().clone()


    def simulate_w_clutch(self,
                          garment_v,
                          garment_f,
                          garment_ref,
                          garment_yms,
                          garment_ts,
                          garment_att,
                          garment_v_init,
                          clutches_v,
                          clutches_v_ref,
                          clutches_f,
                          clutches_yms,
                          clutches_ts,
                          clutches_attach_v,
                          clutches_attach_fids,
                          clutches_attach_bcs,
                          clutches_attach_v_center,
                          clutches_attach_fids_center,
                          clutches_attach_bcs_center,
                          body_v,
                          body_vn,
                          body_v_ref):
        """
        Simulates a garment as it slides over the body
        """

        # Init Energies
        sdf_g = SDF(vertices=body_v, vertex_normals=body_vn)
        cst_g = CSTri(faces=garment_f, vertices_ref=garment_ref)

        sdf_c = SDF(vertices=body_v, vertex_normals=body_vn)
        cst_c = CSTri(faces=clutches_f, vertices_ref=clutches_v_ref)

        # Prime SDF cache with ref vertices of garment and clutches
        # sdf_g.setup_cache(garment_ref, body_v_ref)
        # sdf_c.setup_cache(clutches_v, body_v_ref)

        # Optimization Params
        garment_v = garment_v.requires_grad_(True)
        clutches_v = clutches_v.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [garment_v, clutches_v],
            lr=1,
            tolerance_grad=1e-32,
            tolerance_change=1e-32,
            max_iter=self.lbfgs_iter,
            line_search_fn='strong_wolfe')


        def get_obj_vals():
            g_strain = cst_g(garment_v, youngmoduli=garment_yms, thicknesses=garment_ts)
            g_attach = self.attach(garment_att, garment_f, garment_v, garment_v_init) if torch.any(garment_att) else 0
            g_sdf = sdf_g(garment_v, inside_only=self.sdf_inside_only)

            c_strain = cst_c(clutches_v, youngmoduli=clutches_yms, thicknesses=clutches_ts, relax=False)
            c_attach_anchors = self.attach_c(clutches_v[:, clutches_attach_v],
                                             garment_v,
                                             garment_f[clutches_attach_fids],
                                             clutches_attach_bcs)
            c_attach_all = self.attach_c(clutches_v[:, clutches_attach_v_center], garment_v, garment_f[clutches_attach_fids_center], clutches_attach_bcs_center)
            c_sdf = sdf_c(clutches_v, inside_only=self.sdf_inside_only)

            obj_vals = g_strain * self.w_strain + g_attach * self.w_attach + self.w_sdf * g_sdf \
                    + c_strain * self.w_c_strain + c_attach_anchors * self.w_c_attach \
                       + c_attach_all * self.w_c_all_attach + self.w_c_sdf * c_sdf

            return obj_vals

        def get_obj_val():
            optimizer.zero_grad()
            obj_val = get_obj_vals().sum()
            obj_val.backward()
            # Grad Norm (check for convergence on garment/clutch vertices)
            # print('i-{:3d}. Sim Grad Norm LBFGS Eval: {:.16}'.format(i, torch.norm(garment_v.grad).cpu().numpy()))

            return obj_val

        for i in range(self.outer_iter):
            optimizer.step(get_obj_val)

        return garment_v.detach().clone(), clutches_v.detach().clone()


    def forces(self, garment_v, garment_f, garment_ref, garment_yms, garment_ts, garment_att, garment_v_init, body_v, body_vn):
        """
        Collect forces for each energy. Forces are the gradient of each energy wrt the DOFs (garment vertices in R3).
        """

        # Init Energies
        sdf = SDF(vertices=body_v, vertex_normals=body_vn)
        cst = CSTri(faces=garment_f, vertices_ref=garment_ref)

        # Optimization Params
        garment_v = garment_v.requires_grad_(True)

        cst_forces = torch.autograd.grad(cst(garment_v, youngmoduli=garment_yms, thicknesses=garment_ts).sum() * self.w_strain, garment_v)[0]
        attach_forces = torch.autograd.grad(self.attach(garment_att, garment_f, garment_v, garment_v_init).sum() * self.w_attach, garment_v)[0]
        sdf_forces = torch.autograd.grad(sdf(garment_v, inside_only=True).sum() * self.w_sdf, garment_v)[0]

        return cst_forces.detach(), attach_forces.detach(), sdf_forces.detach()

    def attach(self, att, f, v, v_init):
        """Attach energy"""
        att_i = f[att]
        dist = v[:, att_i].mean(-2) - v_init[:, att_i].mean(-2)
        return torch.pow(dist, 2).sum(-1).mean(-1)

    def attach_c(self, clutch_points, v, f, bcs ):
        """Attach energy"""
        triangles = v[:, f]
        garment_points = barycentric_to_points(triangles=triangles.view((-1, 3, 3)),
                                              bc_coords=bcs.repeat(len(v), 1)).view(len(v), -1, 3)
        return torch.pow(clutch_points - garment_points, 2).sum(-1).mean(-1)

