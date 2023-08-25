import os
from cdkg.renderables.garment import Garment
from cdkg.models.garment_on_body import GarmentOnBody
from cdkg.kg_viewer import KGViewer
from cdkg.renderables.clutches import Clutches
from cdkg.configuration import CONFIG as C
from aitviewer.configuration import CONFIG as AVC
import cdkg.poses as poses

if __name__ == '__main__':
    """
    Simulate a garment on the body, optionally with clutches attached.
    """

    AVC.update_conf(C.conf)

    # Models
    garment_model = GarmentOnBody()
    body = poses.arm_forward()

    # Load Body / Garment / Clutches
    garment = Garment.from_npz(os.path.join(C.datasets.kg, "garments_opt_uist_22/shirt_opt_all.npz"), body)

    # Optional - the garment on body model will automatically use the garment-only simulation if you comment this out.
    clutches = Clutches.from_npz(os.path.join(C.datasets.kg, "clutches/clutches_3_alt.npz"), garment)
    clutches.states[:] = 0.1
    garment.add(clutches)

    # Simulate
    garment_model.forward(garment=garment, body=body)
    body.add_garment(garment)

    # Add to scene and render
    v = KGViewer()
    v.scene.camera.target = garment.vertices[-1].mean(0)
    v.scene.add(body)
    v.run()

