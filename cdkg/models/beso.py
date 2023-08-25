import numpy as np
from tqdm import tqdm
from cdkg.renderables.garment import Garment
from cdkg.renderables.clutches import Clutches

class BESO():
    """
    Maximizes strain energy density of the given garment with an area constraint
    """

    def __init__(self, body, garment, model):
        """
        :param garment: Garment with starting density configuration
        :param body: Body that includes reference and deformed poses
        """
        self.body = body
        self.garment = garment
        self.clutches = garment.clutches
        self.model = model

        # keep track of densities and vertices as ESO progresses
        self.garment_d_progress = []
        self.garment_v_progress = []
        self.clutches_v_progress = []

        self.area_hat = None
        self.sn_prev = None
        self.temporal_smoothing = True

    def obj_max_min(self, sn):
        """Non-normalized objective"""
        # sn = sn[1:]
        n_motions = len(sn)
        e_max = sn[int(n_motions/2):]
        e_min = sn[:int(n_motions/2)]
        e_diff = e_max - e_min
        return e_diff.sum(0)

    def obj_max_min_normed(self, sn):
        """Normalize"""
        # sn = sn[1:]
        n_motions = len(sn)
        e_max = sn[int(n_motions/2):]
        e_min = sn[:int(n_motions/2)]
        e_diff = e_max - e_min
        # Normalize before summing
        e_diff = e_diff / np.linalg.norm(e_diff, axis=-1)[...,np.newaxis]

        return e_diff.sum(0)


    def obj_max_min_normed_by_max(self, sn):
        """Normalize by max"""
        n_motions = len(sn)
        e_max = sn[int(n_motions/2):]
        e_min = sn[:int(n_motions/2)]

        # Normalize by max before summing
        e_diff = (e_max / e_max.max(-1)[...,np.newaxis]) - (e_min / e_max.max(-1)[...,np.newaxis])

        return e_diff.sum(0)


    def obj_multiple(self, sn):
        return sn[1:].mean(0)

    def tpo(self, iter=150, desc="BESO", d_thresh=0.1, asr=0.15, ear=0.015, ar_max=0.015, keep_attachments=False, path=""):
        # :param iter: Max iterations to run for
        # :param d_thresh: Density value for the second less dense material
        # :param asr: Area Star (Target) ratio. I.e. what percent of the original area do we want remaining?
        # :param ear: Evolutionary Area Ratio - by what % should the area change every iteration?
        # :param ar_max: Maximum amount of material that can be added per iteration
        # :return: Optimized density values

        # initialize current areas
        self.area_hat = self.garment.dense_areas[0]

        # target areas (don't change)
        area_star = self.garment.area * asr

        num_elements = int(self.garment.faces.shape[0])

        # Store previous sensitivity number to avoid flickering of add/remove
        self.sn_prev = None

        # Presim
        self.model.forward(self.garment, self.body, outer_iter=16)

        progress_bar = tqdm(total=iter, desc=desc)

        # Generate density variations
        for i in range(iter):

            # FEA 1
            self.model.forward(self.garment, self.body, outer_iter=8)

            self.save_progress(i)

            # Opt connections (objective is max-min)
            self.opt_connections(self.garment, area_star, ear, ar_max, num_elements, d_thresh, keep_attachments, self.obj_max_min_normed_by_max)

            progress_bar.update()
        progress_bar.close()

        # bodies_eso = ['rest', 'hips_ext_left', 'hips_flex_left', 'knee_flex_left', 'hips_ext_right', 'hips_flex_right', 'knee_flex_right']
        bodies_eso = ['rest', 'arm_forward', 'arm_up', 'arm_flex', 'arm_back', 'torso_bend_forward_alt']

        n_motions = int(len(self.garment_v_progress)/2)

        # OFF States
        for i in range(n_motions*2):

            g_name = bodies_eso[i] + '_off' if i<n_motions else  bodies_eso[i - n_motions] + '_on'
            c_name = bodies_eso[i] + '_clutches_off' if i<n_motions else  bodies_eso[i - n_motions] + '_clutches_on'

            g = Garment(vertices=self.garment_v_progress[i],
                      faces=self.garment.faces,
                      att=self.garment.att,
                      densities=self.garment_d_progress[i],
                      name=self.garment.name + "_" + g_name,
                      f_mask=self.garment.f_mask,
                      subdivide=self.garment.subdivide,
                      vertices_ref=self.garment.vertices_ref)

            c = Clutches(
                bc_coords=self.clutches.bc_coords,
                bc_dirs=self.clutches.bc_dirs,
                bc_fids=self.clutches.bc_fids,
                lengths=self.clutches.lengths,
                pre_strains=self.clutches.pre_strains,
                vertices=self.clutches_v_progress[i],
                faces=self.clutches.faces,
                vertices_ref=self.clutches.vertices_ref,
                # states=clutches_config_s[i],
                name=self.garment.name + "_" + c_name
                )

            if i < n_motions:
                c.states[:] = 0.1  #Turn off all clutches for OFF states

            # Save components
            g.to_npz(path=path)
            c.to_npz(path=path)

    def opt_connections(self, garment, area_star, ear, ar_max, num_elements, d_thresh, keep_attachments, obj_fn):

        # If we reach target area, then stop evolving
        if np.abs(self.area_hat - area_star) < 1e-3:
            ear = 0
            # print('Reached target area')

        # Adjust current area
        dir = (1 - ear) if self.area_hat > area_star else (1 + ear)
        # area_hat *= dir
        area_hat_new = self.area_hat * dir  # if area_hat > 0 else area_star*0.25
        delta_area = self.area_hat - area_hat_new
        self.area_hat = area_hat_new

        # print(area_hat / garment.area)
        sn = garment.sensitivities

        if self.temporal_smoothing:
            sn = sn if self.sn_prev is None else (sn + self.sn_prev) * 0.5
            self.sn_prev = garment.sensitivities

        sn = obj_fn(sn)
        # clutch version - OFF for sn[1] on ON for sn[2]. minimize sn[1] maximize sn[2]
        # e_max = sn[2]
        # e_min = sn[1]
        # sn = e_max - e_min

        # Keep attachments
        if keep_attachments:
            sn[garment.att] = 1e15

        # Sort elements by energy density (sensitivity assumption)
        sn_order = (-sn).argsort()  # Most sensitive elements first!!
        sn_sorted = sn[sn_order]

        sensitivity_threshold = sn[0]
        sorted_areas = garment.f_rest_areas[sn_order]
        ta = 0
        for k, a in enumerate(sorted_areas):
            ta += a
            if ta > self.area_hat:
                # Sensitivity threshold is set to the element right before we have satisfied our area budget
                sensitivity_threshold = sn_sorted[k - 1]
                break

        # Elements to Add/Remove (as a mask over all elements)
        soft_element_mask = garment.densities[-1] == d_thresh
        hard_element_mask = ~soft_element_mask
        el_to_add = np.logical_and(sn > sensitivity_threshold, soft_element_mask)
        el_to_rem = np.logical_and(sn < sensitivity_threshold, hard_element_mask)

        # admission ratio
        ar = np.count_nonzero(el_to_add) / num_elements
        # print(ar)
        # Should not exceed max admission ratio to keep stable
        if ar > ar_max:
            max_elem_to_add = int(ar_max * num_elements)
            add_threshold = sn[soft_element_mask][(-sn[soft_element_mask]).argsort()][max_elem_to_add]
            el_to_add = np.logical_and(sn > add_threshold, soft_element_mask)
            area_to_add = garment.f_rest_areas[el_to_add].sum()

            area_to_remove = delta_area + area_to_add
            area_removed = 0
            remove_threshold = 0.0

            for k in range(len(sorted_areas) - 1, -1, -1):
                a = sorted_areas[k]
                # densities not sorted
                if hard_element_mask[sn_order][k]:
                    area_removed += a
                    if area_removed > area_to_remove:
                        # Sensitivity threshold is set to the element right before we have satisfied our area budget
                        remove_threshold = sn_sorted[k]
                        break
            el_to_rem = np.logical_and(sn < remove_threshold, hard_element_mask)

        # Adjust densities
        el_mask = np.arange(0, len(garment.faces))
        el_to_add_mask = el_mask[el_to_add]
        el_to_rem_mask = el_mask[el_to_rem]
        garment.densities[:, el_to_add_mask] = 1.0
        garment.densities[:, el_to_rem_mask] = d_thresh

    def opt_clutches(self, garment, area_star, ear, ar_max, num_elements):
        pass
    
    def save_progress(self, i):

        # Add Rest Frame and Setup Configuration States (combination of motions and clutch states)
        if i == 0:
            self.garment_d_progress = self.garment.densities[[0]].copy()
            self.garment_v_progress = self.garment.vertices[[0]].copy()
            self.garment_d_progress = np.repeat(self.garment_d_progress[np.newaxis], len(self.body.poses_body), axis=0)
            self.garment_v_progress = np.repeat(self.garment_v_progress[np.newaxis], len(self.body.poses_body), axis=0)

            if self.clutches is not None:
                self.clutches_v_progress = self.clutches.vertices[[0]].copy()
                self.clutches_v_progress = np.repeat(self.clutches_v_progress[np.newaxis], len(self.body.poses_body), axis=0)

        self.garment_d_progress = np.concatenate((self.garment_d_progress, self.garment.densities[:, np.newaxis]), axis=1)
        self.garment_v_progress = np.concatenate((self.garment_v_progress, self.garment.vertices[:, np.newaxis]), axis=1)

        if self.clutches is not None:
            self.clutches_v_progress = np.concatenate((self.clutches_v_progress, self.clutches.vertices[:, np.newaxis]), axis=1)
