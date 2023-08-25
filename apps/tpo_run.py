import os
from cdkg.renderables.garment import Garment
from cdkg.renderables.body import Body
from aitviewer.models.star import STARLayer
from cdkg.models.beso import BESO
from cdkg.configuration import CONFIG as C
from cdkg.models.garment_on_body import GarmentOnBody
from cdkg.renderables.clutches import Clutches
import torch
import cdkg.poses as poses


if __name__ == '__main__':
    """Runs On-Body Topology Optimization"""

    # Models
    garment_on_body = GarmentOnBody(w_attach=0.003)
    star_layer = STARLayer(device=C.device)

    # Poses/Garments
    body = poses.poses_upper_body()
    n_motions = len(body.poses_body)

    # Repeat all poses for clutch state off
    body = Body(
        poses_body=torch.cat((body.poses_body, body.poses_body)),
        poses_root=torch.cat((body.poses_root, body.poses_root)),
        smpl_layer=star_layer,
        name=body.name
    )

    garment = Garment.from_npz(os.path.join(C.datasets.kg, "garments/shirt_long.npz"), body)
    clutches = Clutches.from_npz(os.path.join(C.datasets.kg, "clutches/shirt_long_clutches_6.npz"), garment)
    clutches.states[:n_motions] = 0.1
    garment.add(clutches)

    # Prep
    experiment_name = garment.name + "_" + clutches.name + "_asr" + C.tpo.asr
    print("Experiment: " + experiment_name)

    # Create experiment dir if not exists
    path = "../data/eso/out/" + experiment_name + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    # BESO - Will save directly to disk
    beso = BESO(body, garment, garment_on_body)
    beso.tpo(desc=experiment_name, path=path, asr=C.tpo.asr, iter=C.tpo.beso_iter)
