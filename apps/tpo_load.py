from cdkg.renderables.garment import Garment
from cdkg.renderables.clutches import Clutches
from cdkg.models.garment_on_body import GarmentOnBody
from cdkg.kg_viewer import KGViewer
from aitviewer.models.star import STARLayer
from cdkg.configuration import CONFIG as C
import cdkg.poses as poses

if __name__ == '__main__':

    # Models
    garment_model = GarmentOnBody()
    star_layer = STARLayer(device=C.device)

    motions = ['arm_flex']

    path = '../data/eso/uist22/sleeve_arm_clutches_2_alt_w001_all/'
    garment = 'sleeve_arm'
    bodies = []

    for i, m in enumerate(motions):
        # Motions
        body = poses.from_pose(pose=motions[i])

        # Garments
        garment_off = Garment.from_npz(path + garment + '_' + motions[i] + '_off.npz')
        garment_on = Garment.from_npz(path + garment + '_' + motions[i] + '_on.npz')

        # Clutches
        clutch_off = Clutches.from_npz(path + garment + '_' + motions[i] + '_clutches_off.npz')
        clutch_off.states[:] = 0.1
        clutch_off.forward_energy()
        clutch_on = Clutches.from_npz(path + garment + '_' + motions[i] + '_clutches_on.npz')

        # Optionally First Sim the Motion
        # body_seq = Body(poses_body=poses_to_seq(body.poses_body, 20), poses_root=poses_to_seq(body.poses_root, 20), smpl_layer=star_layer, name=body.name + "_seq")
        # garment_full = Garment.from_npz('../data/garments/'+garment+'.npz', body=body_seq)
        # clutch_full = Clutches.from_npz('../data/clutches/clutches_2_alt.npz', garment_full)
        # clutch_full.states[:] = 0.1
        # garment_full.add(clutch_full)
        # garment_model.forward(garment=garment_full, body=body_seq)
        # garment_off.vertices = np.concatenate((garment_full.vertices, garment_off.vertices[1:]), axis=0)
        # garment_off.densities = np.concatenate((garment_full.densities, garment_off.densities[1:]), axis=0)
        # clutch_off.vertices = np.concatenate((clutch_full.vertices, clutch_off.vertices[1:]), axis=0)
        # clutch_off.states = np.concatenate((clutch_full.states, clutch_off.states[1:]), axis=0)
        # clutch_off.n_frames = clutch_off.n_frames + clutch_full.n_frames - 1
        # garment_off.forward_energy()
        # clutch_off.cst_model = clutch_off.get_cst_model()
        # clutch_off.forward_energy()
        # body = Body(poses_body=torch.cat((body_seq.poses_body, body.poses_body[1:]), dim=0), poses_root=torch.cat((body_seq.poses_root, body.poses_root[1:]), dim=0), smpl_layer=star_layer, name=body.name)

        # Optionally Plot
        # plot_max_min(garment_on,
        #              garment_off,
        #              clutch_on=clutch_on,
        #              clutch_off=clutch_off,
        #              title=motions[i])

        # Add to viewer
        garment_on.add(clutch_on)
        body.add_garment(garment_on)
        garment_off.add(clutch_off)
        body.add_garment(garment_off)
        body.position += (i, 0.0, 0.0)
        bodies.append(body)

    v = KGViewer()
    v.scene.add(*bodies)
    v.run()
