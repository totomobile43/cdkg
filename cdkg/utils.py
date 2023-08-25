import trimesh
import numpy as np
import torch
import networkx as nx
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aitviewer.renderables.meshes import Meshes as AVMeshes
from cdkg.configuration import CONFIG as C

c2c = lambda t: t.detach().cpu().numpy()
to_n = lambda t: t.detach().cpu().numpy() if torch.is_tensor(t) else t
to_t = lambda n: torch.from_numpy(n).to(device=C.device) if not torch.is_tensor(n) else n

def poses_to_seq(poses, frames, deccelerate=False):
    steps = torch.arange(0, frames, 1, device=poses.device).unsqueeze(-1)
    if deccelerate:
        steps = steps **0.5
        steps = (steps / steps.max()) * frames

    p = ((poses[1] - poses[0]) / (frames-1)).unsqueeze(0)
    return steps*p + poses[[0]]

def load_amass_pose(npz_data_path, start_frame=None, end_frame=None):
    """Load a sequence downloaded from the AMASS website."""
    body_data = np.load(npz_data_path)

    sf = start_frame or 0
    ef = end_frame or body_data['poses'].shape[0]
    poses = body_data['poses'][sf:ef]
    num_body_joints = 23
    i_root_end = 3
    i_body_end = i_root_end + num_body_joints * 3
    poses_root = to_t(poses[:, :i_root_end]).to(device=C.device, dtype=C.f_precision)
    poses_body = to_t(poses[:, i_root_end:i_body_end]).to(device=C.device, dtype=C.f_precision)
    return poses_body, poses_root

def mold_body(body, garment, indent=0.003):

    # Get v-mask
    body_v, body_f = body.vertices.copy(), body.faces.copy()
    if garment.subdivide:
        body_v, body_f = subdivide_meshes(body_v, body_f, num_subdivisions=garment.subdivide)
    faces_masked = body_f[garment.f_mask]
    v_mask = np.unique(faces_masked.reshape(-1))

    n_vertices = body.vertices.shape[1]
    n_frames = garment.vertices.shape[0]
    v = body.vertices.copy()
    vn = body.vertex_normals
    # Repeat last frame of vertex normals of body until it fits lengths of vertices in garment
    if len(v) < n_frames:
        v = np.concatenate((v, v[[-1]].repeat(n_frames-len(v),0)))
        vn = np.concatenate((vn, vn[[-1]].repeat(n_frames-len(vn),0)))
    rvmask = v_mask[v_mask < n_vertices]
    v[:, rvmask] -= vn[:, rvmask] * indent

    body = AVMeshes(v, body.faces, name=body.name, color=body.mesh_seq.color)
    return body

def subdivide_meshes_torch(vertices, faces, num_subdivisions=1):
    """
    Subdivide using pytorch3D utilities
    """
    m = Meshes(verts=vertices, faces=faces.unsqueeze(0).repeat(len(vertices), 1, 1))
    for n in range(num_subdivisions):
        m = SubdivideMeshes()(m)
    v = m.verts_padded()
    f = m.faces_padded()[0]
    return v, f

def subdivide_meshes(vertices, faces, num_subdivisions=1):
    """
    Subdivide using pytorch3D utilities
    """
    f = to_t(faces)
    v = to_t(vertices)
    m = Meshes(verts=v, faces=f.unsqueeze(0).repeat(len(v), 1, 1))
    for n in range(num_subdivisions):
        m = SubdivideMeshes()(m)
    v = to_n(m.verts_padded())
    f = to_n(m.faces_padded()[0])
    return v, f


def plot_data(garment, normalize=True, title="BESO Opt"):

    # Remove rest frame
    total_energy = garment.dense_energies[1:]
    total_areas = garment.dense_areas[1:]

    # Normalize by area
    total_energy_density = total_energy / total_areas
    area_percent = (garment.dense_areas / garment.area)[1:]

    # Normalize both
    if normalize:
        total_energy_density = total_energy_density / total_energy_density[[0]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Iteration")
    # Set y-axes titles
    fig.update_yaxes(title_text="Relative <b>Coverage</b>", secondary_y=False)
    fig.update_yaxes(title_text="Relative <b>Energy Density</b>", secondary_y=True)

    # Add figure title
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        yaxis=dict(showgrid=False, showline=False),
        xaxis = dict(showgrid=False, showline=False)
    )
    fig['layout']['yaxis2']['showgrid'] = False

    fig.add_trace(go.Scatter(
        y=total_energy_density,
        mode='lines',
        name='Relative Energy Density'
    ), secondary_y = True)

    fig.add_trace(go.Scatter(
        y=area_percent,
        mode='none',
        fill='tonexty',
        name='Coverage %'
    ))
    fig.show()


def plot_max_min(garment_on,
                 garment_off,
                 clutch_on=None,
                 clutch_off=None,
                 normalize=True,
                 title="BESO Opt"):

    # Remove rest frame
    total_energy_on = garment_on.dense_energies[1:]
    total_energy_off = garment_off.dense_energies[1:]

    total_areas = garment_on.dense_areas[1:]

    if clutch_on is not None:
        total_energy_on += clutch_on.energies[1:]
        total_areas +=clutch_on.area
    if clutch_off is not None:
        total_energy_off += clutch_off.energies[1:]

    # Normalize by area
    total_energy_density_on = total_energy_on / total_areas
    total_energy_density_off = total_energy_off / total_areas

    area_percent = (garment_on.dense_areas / garment_on.area)[1:]


    # Normalize both
    if normalize:
        total_energy_density_on = total_energy_density_on / total_energy_density_on[[0]]
        total_energy_density_off = total_energy_density_off / total_energy_density_off[[0]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Iteration")
    # Set y-axes titles
    fig.update_yaxes(title_text="Relative <b>Coverage</b>", secondary_y=False)
    fig.update_yaxes(title_text="Relative <b>Energy Density</b>", secondary_y=True)

    # Add figure title
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title
    )

    fig.add_trace(go.Scatter(
        y=total_energy_density_on,
        mode='lines',
        name='Clutches ON'
    ), secondary_y = True)

    fig.add_trace(go.Scatter(
        y=total_energy_density_off,
        mode='lines',
        name='Clutches OFF'
    ), secondary_y = True)

    fig.add_trace(go.Scatter(
        y=area_percent,
        mode='none',
        fill='tonexty',
        name='Coverage %'
    ))
    fig.show()

def plot_energy(energy, energy2, energy3, energy4, energy5):

    energy = np.concatenate((energy[10:20][::-1], energy[1:10]), axis=-1)
    energy2 = np.concatenate((energy2[10:20][::-1], energy2[1:10]), axis=-1)
    energy3 = np.concatenate((energy3[10:20][::-1], energy3[1:10]), axis=-1)
    energy4 = np.concatenate((energy4[10:20][::-1], energy4[1:10]), axis=-1)
    energy5 = np.concatenate((energy5[10:20][::-1], energy5[1:10]), axis=-1)

    fig = make_subplots()
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Theta")
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Energy</b>")
    # fig.update_yaxes(title_text="<b>Area %</b>")

    # Add figure title
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.add_trace(go.Scatter(
        y=energy,
        mode='lines',
        name='Opt_for_both_dir_area_15'
    ))

    fig.add_trace(go.Scatter(
        y=energy2,
        mode='lines',
        name='Opt_for_both_dir_area_20'
    ))

    fig.add_trace(go.Scatter(
        y=energy3,
        mode='lines',
        name='Opt_for_both_dir_area_25'
    ))

    fig.add_trace(go.Scatter(
        y=energy4,
        mode='lines',
        name='Opt_for_ext_area_15'
    ))

    fig.add_trace(go.Scatter(
        y=energy5,
        mode='lines',
        name='Opt_for_flex_area_15'
    ))

    fig.show()


def get_masked_mesh(v_mask, vertices, faces):
    # Construct indices of the masked vertices
    v_indices = torch.arange(vertices.shape[1])[v_mask].to(device=C.device, dtype=C.i_precision)

    # Pairwise comparison between each face's vertex indices and the indices in the v_mask
    f_mask = torch.any(faces.unsqueeze(1) == v_indices.repeat((3, 1)).T, dim=-2)
    # Faces that include all 3 vertices
    f_mask_all = torch.all(f_mask, dim=-1)

    # Map of original vertex index to new vertex index
    v_to_masked_v = torch.zeros(vertices.shape[1]).to(device=C.device, dtype=C.i_precision)
    v_to_masked_v[v_mask] = torch.arange(v_indices.shape[0]).to(device=C.device, dtype=C.i_precision)

    vertices_masked = vertices[:, v_mask]  # Select subset of vertices
    faces_masked = faces[f_mask_all]  # Select subset of faces
    faces_masked = v_to_masked_v[faces_masked]  # Adjust their indices

    return vertices_masked, faces_masked


def get_identity_quaternion_with_eps(shape=(1, 4), eps=1e-24):
    iq = torch.zeros(shape).to(dtype=C.f_precision, device=C.device)
    iq[:, 0] = iq[:, 0] + (1 - eps)
    iq[:, 1] = iq[:, 1] + eps
    iq = iq / torch.norm(iq, dim=-1).unsqueeze(-1)
    return iq


def barycentric_implicit_to_explicit(bc_coords):
    return torch.cat((bc_coords, (1 - bc_coords.sum(-1)).unsqueeze(-1)), dim=-1)

# def barycentric_implicit_to_explicit_n(bc_coords):
#     return np.concatenate((bc_coords, (1 - bc_coords.sum(-1))[np.newaxis]), axis=-1)


def barycentric_explicit_to_implicit(bc_coords):
    return bc_coords[..., :-1]


def barycentric_to_points(triangles, bc_coords):
    """
    Convert a list of barycentric coordinates on a list of triangles
    to cartesian points.

    **Adapated to Pytorch from Trimesh implementation**

    Parameters
    ------------
    triangles : (n, 3, 3) float
      Triangles in space (Tensor)
    bc_coords : (n, 2) float
      Barycentric coordinates (tensor)

    Returns
    -----------
    points : (m, 3) float
      Points in space
    """

    # Triangles is only a single vector, needs to be wrapped in array
    if not trimesh.util.is_shape(triangles, (-1, 3, 3)):
        triangles = triangles.unsqueeze(0) #torch.ones((1, len(triangles), 3), dtype=bc_coords.dtype, device=bc_coords.device) * triangles
        # raise ValueError('Triangles must be (n,3,3)!')

    # Barycentric is only a single vector, needs to be wrapped in an array
    if bc_coords.shape == (2,):
        bc_coords = bc_coords.unsqueeze(0) # torch.ones((len(triangles), 2), dtype=bc_coords.dtype, device=bc_coords.device) * bc_coords
    elif bc_coords.shape == (3,):
        bc_coords = bc_coords.unsqueeze(0) #torch.ones((len(triangles), 3), dtype=bc_coords.dtype, device=bc_coords.device) * bc_coords

    # Convert implicit barycentric (u,v) to explicit (w,u,v)
    if trimesh.util.is_shape(bc_coords, (len(triangles), 2)):
        bc_coords = barycentric_implicit_to_explicit(bc_coords)

    elif not trimesh.util.is_shape(bc_coords, (len(triangles), 3)):
        raise ValueError('Barycentric shape incorrect!')

    t_barycentric_points = bc_coords / bc_coords.sum(dim=1).view((-1, 1))
    t_points = (triangles * t_barycentric_points.view(-1, 3, 1)).sum(dim=1)

    return t_points


def points_to_barycentric(triangles,
                          points,
                          method='cross'):
    """
    Find the barycentric coordinates of points relative to triangles.
    Adapted from Trimesh for pytorch.

    **Adapated to Pytorch from Trimesh implementation**

    The Cramer's rule solution implements:
        http://blackpawn.com/texts/pointinpoly

    The cross product solution implements:
        https://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf


    Parameters
    -----------
    triangles : (n, 3, 3) float
      Triangles vertices in space
    points : (n, 3) float
      Point in space associated with a triangle
    method :  str
      Which method to compute the barycentric coordinates with:
        - 'cross': uses a method using cross products, roughly 2x slower but
                  different numerical robustness properties
        - anything else: uses a cramer's rule solution

    Returns
    -----------
    barycentric : (n, 3) float
      Barycentric coordinates of each point
    """

    def method_cross():
        n = torch.cross(edge_vectors[:, 0], edge_vectors[:, 1])
        denominator = diagonal_dot(n, n)

        barycentric = torch.zeros((len(triangles), 3), dtype=C.f_precision, device=C.device)
        barycentric[:, 2] = diagonal_dot(torch.cross(edge_vectors[:, 0], w), n) / denominator
        barycentric[:, 1] = diagonal_dot(torch.cross(w, edge_vectors[:, 1]), n) / denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
        return barycentric

    def method_cramer():
        dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
        dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
        dot02 = diagonal_dot(edge_vectors[:, 0], w)
        dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
        dot12 = diagonal_dot(edge_vectors[:, 1], w)

        inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01)

        barycentric = torch.zeros((len(triangles), 3), device=C.device, dtype=C.f_precision)
        barycentric[:, 2] = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
        barycentric[:, 1] = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
        return barycentric

    # Make sure we are using tensors
    if not torch.is_tensor(triangles):
        triangles = torch.tensor(triangles)

    if not torch.is_tensor(points):
        points = torch.tensor(points)

    # establish that input triangles and points are sane
    if not trimesh.util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError('triangles shape incorrect')
    if not trimesh.util.is_shape(points, (len(triangles), 3)):
        raise ValueError('triangles and points must correspond')

    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = points - triangles[:, 0].view((-1, 3))

    if method == 'cross':
        return method_cross()
    return method_cramer()


def diagonal_dot(a, b):
    """
    Dot product by row of a and b.

    Parameters
    ------------
    a : (m, d) float
      First array
    b : (m, d) float
      Second array

    Returns
    -------------
    result : (m,) float
      Dot product of each row
    """
    # 3x faster than (a * b).sum(axis=1)
    # avoiding np.ones saves 5-10% sometimes
    return (a * b).sum(dim=-1)
    # return torch.dot(a * b, torch[1.0] * a.shape[1])


def rodrigues_rotation(a, b, v):
    """
    Apply rotation from a->b to vector v using rodrigues formula
    Args:
        v: input vector to rotate,
        a & b: vectors defining the plane of rotation

    Returns:
      vec: rotated vector
    """

    c = torch.cross(a, b)
    sin_theta = torch.norm(c)
    cos_theta = torch.dot(a, b)
    rot_axis = c / sin_theta

    if sin_theta == 0:
        return v
    else:
        return v * cos_theta - torch.cross(v, rot_axis) * sin_theta + rot_axis * torch.dot(v, rot_axis) * (1.0 - cos_theta)


def rodrigues_rotation_np(a, b, v):
    """
    Apply rotation from a->b to vector v using rodrigues formula
    Args:
        v: input vector to rotate,
        a & b: vectors defining the plane of rotation

    Returns:
      vec: rotated vector
    """

    c = np.cross(a, b)
    sin_theta = np.linalg.norm(c)
    cos_theta = np.dot(a, b)
    rot_axis = c / sin_theta

    if sin_theta == 0:
        return v
    else:
        return v * cos_theta - np.cross(v, rot_axis) * sin_theta + rot_axis * np.dot(v, rot_axis) * (1.0 - cos_theta)


def djikstra_path(vertices, faces, face_ids, edges):
    """ Compute Djikstra shortest path between nodes/anchors of edges """

    # Setup mesh/graph structure
    g = nx.Graph()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    for e, l in zip(mesh.edges_unique, mesh.edges_unique_length): g.add_edge(*e, length=l)

    # Get the vertex ids for each face id in the eddges. Arbirtrarily select the first vertex of a face
    node_v_ids = mesh.faces[face_ids[edges]][:, :, 0]

    # Get shorterst path and trim to max ppe (points per edge)
    paths = [np.array(nx.shortest_path(g, nv[0], nv[1], weight='length')) for nv in node_v_ids]

    # Points per edge should be proportional to the number of vertices in the path
    ppe = max([len(p) for p in paths])
    # ppe = max(ppe, min_points)

    # Some paths may be too short (jagged array), pad them with the last element of the array
    paths = np.array([np.pad(p, (0, ppe - len(p)), mode='edge') if ppe - len(p) > 0 else p for p in paths])

    # Place path onto triangles
    vertex_face_ids = mesh.vertex_faces[paths][:, :, 0]
    vertex_bcs_impl = np.random.rand(*vertex_face_ids.shape)[..., np.newaxis].repeat(2, -1) / 2
    vertex_bcs_expl = np.concatenate((vertex_bcs_impl, (1 - vertex_bcs_impl.sum(-1))[..., np.newaxis]), axis=-1)

    return vertex_bcs_expl, vertex_face_ids


@torch.jit.script
def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30)
    return res


def gradient_check(obj_fx, obj_val, opt_params, *obj_fn_args, epsilon=1e-5):
    with torch.no_grad():

        opt_grad = opt_params.grad.clone()
        opt_grad_unrolled = opt_grad.view(-1)
        opt_params = opt_params.detach().clone()
        opt_params_unrolled = opt_params.view(-1)

        d = torch.ones_like(opt_params_unrolled)
        d = d / torch.norm(d)

        residual_saved = torch.zeros_like(opt_grad_unrolled)

        for n in range(30):
            d_epsilon = d * epsilon
            residual = torch.zeros_like(opt_grad_unrolled)

            for i in range(len(opt_params_unrolled)):
                # Obj Value
                test_opt_params = opt_params_unrolled.clone()
                test_opt_params[i] += d_epsilon[i]
                obj_val_eps = obj_fx(test_opt_params.view(opt_params.shape), *obj_fn_args)

                # Est. Obj Value from Gradient
                obj_val_eps_from_gradient = opt_grad_unrolled[i] * d_epsilon[i] + obj_val

                # Compute residual
                residual[i] = torch.abs(obj_val_eps - obj_val_eps_from_gradient)

            if n > 0:
                residual_ratio = residual / residual_saved
                print("Residual ratio with epsilon {:.16f} is max {:.8f}, min {:.8f}, and mean {:.8f}  ".format(epsilon,
                                                                                                                residual_ratio.max(),
                                                                                                                residual_ratio.min(),
                                                                                                                residual_ratio.mean()))

            residual_saved = residual
            epsilon = epsilon * 0.5

def clutch_model(bc_coord, bc_dir, bc_fid, length, s_vertices, s_faces, s_normals):
    curve_points, curve_normals, curve_bcs, curve_fids, curves_points_side = bcs_to_curves(bc_coord, bc_dir, bc_fid, length, s_vertices, s_faces, s_normals)
    vertices, faces = curves_to_meshes(curve_points, curve_normals)
    v_offset = curve_points.shape[1]
    vertices[:, v_offset] = curves_points_side[:, 0]
    vertices[:, v_offset*2] = curves_points_side[:, 1]
    vertices[:, v_offset*2 - 1] = curves_points_side[:, 2]
    vertices[:, v_offset*3 - 1] = curves_points_side[:, 3]

    return curve_points, curve_bcs, curve_fids, vertices, faces


def curves_to_meshes(curves_points, curve_normals, width=0.005):

    # Project vertices orthonormal to the curve and curve normal, i.e. left and right of the curve
    v_center = curves_points
    curve_tangents = v_center[:, :-1] - v_center[:, 1:]
    curve_tangents = torch.cat((curve_tangents, curve_tangents[:, [-1]]), dim=1) #repeat last tangent
    curve_tangents /= torch.linalg.norm(curve_tangents, dim=-1).unsqueeze(-1)
    ortho_normals = torch.cross(curve_tangents, curve_normals, dim=-1)
    v_left = curves_points + ortho_normals*width
    v_right = curves_points - ortho_normals*width
    v_all = torch.cat((v_center, v_left, v_right), dim=1) #+ curve_normals.repeat((1,3,1))*.001
    vertices = v_all

    # Construct faces (mesh topology)
    v_offset = v_center.shape[1]
    t1_left = torch.tensor((0, v_offset+1, v_offset), device=C.device, dtype=C.i_precision)
    t2_left = torch.tensor((0, 1, v_offset + 1), device=C.device, dtype=C.i_precision)
    t1_right = torch.tensor((0, v_offset*2, 1), device=C.device, dtype=C.i_precision)
    t2_right = torch.tensor((v_offset*2, v_offset*2 + 1, 1), device=C.device, dtype=C.i_precision)
    single_segment = torch.stack((t1_left, t1_right, t2_left, t2_right))
    repeat_vec = torch.arange(0, v_offset-1, device=C.device, dtype=C.i_precision)
    # faces = single_segment
    faces = (repeat_vec.unsqueeze(-1).unsqueeze(-1) + single_segment.unsqueeze(0)).view(-1, 3)

    return vertices, faces


def bcs_to_curves(bc_coord, bc_dir, bc_fid, length, v, f, fn, min_seg_legth=0.001):
    """ Take a set of barycentric coordinates, directions and a mesh topology and return curves in world space"""

    # Use v[0] since we always want to walk the surface on the rest state mesh (that's where it was mounted)
    curve_bcs, curve_fids = walk_point(bc_coord, bc_fid, bc_dir, length, v[0], f, fn[0])
    triangles = v[:, f[curve_fids]]
    curves_points = barycentric_to_points(triangles=triangles.view((-1, 3, 3)), bc_coords=curve_bcs.repeat(len(v), 1)).view(len(v), -1, 3)
    curve_normals = fn[:, curve_fids]


    # Remove line segments that are too short
    d_mask = torch.linalg.norm(curves_points[0, :-1] - curves_points[0, 1:], dim=-1) > min_seg_legth
    if torch.any(d_mask):
        d_mask = torch.cat((d_mask, torch.tensor(True, device=bc_coord.device, dtype=torch.bool).unsqueeze(0)))
        d_mask[0] = torch.tensor(True, device=bc_coord.device, dtype=torch.bool)
        curves_points = curves_points[:, d_mask]
        curve_normals = curve_normals[:, d_mask]
        curve_bcs = curve_bcs[d_mask]
        curve_fids = curve_fids[d_mask]

        # return curves_points[:, d_mask], curve_normals[:, d_mask], curve_bcs[d_mask], curve_fids[d_mask]


    curve_tangents = curves_points[:, :-1] - curves_points[:, 1:]
    curve_tangents = torch.cat((curve_tangents, curve_tangents[:, [-1]]), dim=1)  # repeat last tangent
    curve_tangents /= torch.linalg.norm(curve_tangents, dim=-1).unsqueeze(-1)
    curve_orthnormals = torch.cross(curve_tangents, curve_normals, dim=-1)

    # Left Side Start
    target_bc = points_to_barycentric(points=(curves_points[0, 0] + curve_orthnormals[0, 0] * 0.005).unsqueeze(0), triangles=triangles[0, 0].unsqueeze(0))[0]
    c_bcs, c_fids = walk_point(barycentric_explicit_to_implicit(curve_bcs[0]), curve_fids[0], barycentric_explicit_to_implicit(target_bc), 0.005, v[0], f, fn[0])
    curve_bcs_side = c_bcs[[-1]]
    curve_fids_side = c_fids[[-1]]

    # Right Side Start
    target_bc = points_to_barycentric(points=(curves_points[0, 0] - curve_orthnormals[0, 0] * 0.005).unsqueeze(0), triangles=triangles[0, 0].unsqueeze(0))[0]
    c_bcs, c_fids = walk_point(barycentric_explicit_to_implicit(curve_bcs[0]), curve_fids[0], barycentric_explicit_to_implicit(target_bc), 0.005, v[0], f, fn[0])
    curve_bcs_side = torch.cat((curve_bcs_side, c_bcs[[-1]]), dim=0)
    curve_fids_side = torch.cat((curve_fids_side, c_fids[[-1]]), dim=0)

    # Left Side End
    target_bc = points_to_barycentric(points=(curves_points[0, -1] + curve_orthnormals[0, -1] * 0.005).unsqueeze(0), triangles=triangles[0, -1].unsqueeze(0))[0]
    c_bcs, c_fids = walk_point(barycentric_explicit_to_implicit(curve_bcs[-1]), curve_fids[-1], barycentric_explicit_to_implicit(target_bc), 0.005, v[0], f, fn[0])
    curve_bcs_side = torch.cat((curve_bcs_side, c_bcs[[-1]]), dim=0)
    curve_fids_side = torch.cat((curve_fids_side, c_fids[[-1]]), dim=0)

    # Right Side End
    target_bc = points_to_barycentric(points=(curves_points[0, -1] - curve_orthnormals[0, -1] * 0.005).unsqueeze(0), triangles=triangles[0, -1].unsqueeze(0))[0]
    c_bcs, c_fids = walk_point(barycentric_explicit_to_implicit(curve_bcs[-1]), curve_fids[-1], barycentric_explicit_to_implicit(target_bc), 0.005, v[0], f, fn[0])
    curve_bcs_side = torch.cat((curve_bcs_side, c_bcs[[-1]]), dim=0)
    curve_fids_side = torch.cat((curve_fids_side, c_fids[[-1]]), dim=0)

    triangles_side = v[:, f[curve_fids_side]]
    curves_points_side = barycentric_to_points(triangles=triangles_side.view((-1, 3, 3)), bc_coords=curve_bcs_side.repeat(len(v), 1)).view(len(v), -1, 3)

    curve_bcs = torch.cat((curve_bcs, curve_bcs_side), dim=0)
    curve_fids = torch.cat((curve_fids, curve_fids_side), dim=0)

    return curves_points, curve_normals, curve_bcs, curve_fids, curves_points_side


def walk_point(bc, face, dir, length, v, f, fn):
    """
    Computes a (geodesic) curve  on the surface of a mesh along a given direction d and length l
    Based on https://math.stackexchange.com/questions/2292895/walking-on-the-surface-of-a-triangular-mesh
    """

    assert bc.shape[0] == 2, "Barycentric coordinates must be in implicit form (w,u,v) -> (u,v)"


    curr_bc = barycentric_implicit_to_explicit(bc)
    curr_face = face
    faces = torch.tensor(curr_face, device=bc.device, dtype=C.i_precision).unsqueeze(0)
    bcs = curr_bc.unsqueeze(0)
    t_inf = torch.tensor(1e16, device=bc.device, dtype=bc.dtype)

    # Normalize direction and convert to explicit coords
    dir_bc = barycentric_implicit_to_explicit(dir)
    # dir_bc = dir_bc / torch.linalg.norm(dir_bc)

    # Convert direction and length to target in starting bc coordinate
    tri_curr_face = v[f[curr_face]].unsqueeze(0)
    start_point = barycentric_to_points(bc_coords=curr_bc, triangles=tri_curr_face)
    end_point = barycentric_to_points(bc_coords=dir_bc, triangles=tri_curr_face)
    dir_world = end_point - start_point
    dir_world = dir_world / torch.linalg.norm(dir_world)
    target_bc = points_to_barycentric(points=(start_point + dir_world * length), triangles=tri_curr_face)[0]


    # W coordinate will always be negative as long as we are outside the current triangle
    while torch.sum(target_bc < 0) > 0:

        # The vector to traverse in barycentric coordinates
        vec_bc = target_bc - curr_bc

        # Signed distance to each of the edges
        edge_dists = -(curr_bc / vec_bc)

        # if we are starting on an edge
        if len(faces) > 1:
            curr_edge = edge_dists.abs().argmin()
            edge_dists[curr_edge] = t_inf
        # negative values move away from edge
        edge_dists[edge_dists < 0] = t_inf

        # Smallest positive signed distance is the closest edge. If we are very close to 0, we are at the starting edge
        closest_edge_index = edge_dists.argmin()

        # Edge we will cross next
        edge = f[curr_face][f[curr_face] != f[curr_face][closest_edge_index]]

        # Faces with this edge
        edge_faces = ((f == edge[0]).sum(dim=1) & (f == edge[1]).sum(dim=1)).nonzero()

        # Face we will cross next
        next_faces = edge_faces[edge_faces != curr_face]
        if len(next_faces) > 0:
            next_face = next_faces[0]
        else:
            # Edge of the mesh
            break

        v_curr_face = f[curr_face]
        v_next_face = f[next_face]
        tri_curr_face = v[v_curr_face].unsqueeze(0)
        tri_next_face = v[v_next_face].unsqueeze(0)

        # Intersection point in terms of barycentric coordinates in the original triangle
        bc_coord_intersection = curr_bc + vec_bc * edge_dists[closest_edge_index]

        # Convert barycentric point to world and back to barycentric in the next face's coordinates
        intersection_point = barycentric_to_points(bc_coords=bc_coord_intersection, triangles=tri_curr_face)

        # Mapping from one face to the next
        # Method 1: Re-arrange bcs according to new triangle layout (More exact mapping)
        # v0 = v_next_face[0] == v_curr_face
        # v1 = v_next_face[1] == v_curr_face
        # v2 = v_next_face[2] == v_curr_face
        # next_bc = torch.zeros_like(curr_bc)
        # if torch.any(v0):
        #     next_bc[0] = bc_coord_intersection[v0]
        # if torch.any(v1):
        #     next_bc[1] = bc_coord_intersection[v1]
        # if torch.any(v2):
        #     next_bc[2] = bc_coord_intersection[v2]
        # Method 2: convert intersection point back to bc
        next_bc = points_to_barycentric(triangles=tri_next_face, points=intersection_point)[0]

        # Compute remaining walk vector in world space and rotate it to lie on the next face
        vec_world = barycentric_to_points(bc_coords=target_bc, triangles=tri_curr_face) - intersection_point
        vec_world_rotated = rodrigues_rotation(a=fn[curr_face], b=fn[next_face], v=vec_world[0])

        # Compute the end of the vector in terms of the next faces barycentric coordinates
        target_bc = points_to_barycentric(tri_next_face, intersection_point + vec_world_rotated)[0]

        # Assign for next loop
        curr_bc = next_bc
        curr_face = next_face
        bcs = torch.cat((bcs, curr_bc.unsqueeze(0)))
        faces = torch.cat((faces, curr_face.unsqueeze(0)))

    bcs = torch.cat((bcs, target_bc.unsqueeze(0)))
    faces = torch.cat((faces, curr_face.unsqueeze(0)))

    return bcs, faces
