from cdkg.renderables.garment import Garment
from cdkg.renderables.clutches import Clutches
from cdkg.kg_viewer import KGViewer
from aitviewer.configuration import CONFIG as AVC
from cdkg.configuration import CONFIG as C
import cdkg.poses as poses
import os

if __name__ == '__main__':
    """
    Design garments. Existing designs can be modified by selecting the garment, 
    then using the mode in the GUI to add attachments to body, change density, and adjust clutches.
    """

    AVC.update_conf(C.conf)
    body = poses.poses_upper_body()

    # Modify an Existing Garment
    garment = Garment.from_npz(os.path.join(C.datasets.kg, "garments_opt_uist_22/shirt_hr_arm_raise_opt.npz"), body)
    body.add_garment(garment)

    # Optionally add clutches to modify
    # clutches = Clutches.from_npz(os.path.join(C.datasets.kg, "clutches/clutches_3.npz"), garment)
    # garment.add(clutches)


    v = KGViewer()
    v.scene.add(body)
    v.run()