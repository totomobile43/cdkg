import numpy as np
from aitviewer.utils import to_numpy
from cdkg.renderables.body import Body
import os
from cdkg.configuration import CONFIG as C
from aitviewer.models.star import STARLayer



def rest():
    return Body.a_pose(frames=2)

def push():
    a_pose = Body.a_pose(frames=2)
    # Arm forward
    a_pose.poses_body[1, 46] -= 0.4
    a_pose.poses_body[1, 49] += 0.4
    a_pose.poses_body[1, 37] -= 0.6
    a_pose.poses_body[1, 40] += 0.6

    # Elbow straight
    a_pose.poses_body[1, 52] = 0.0
    a_pose.poses_body[1, 55] = 0.0

    # Arm up slightly
    a_pose.poses_body[1, 47] += 0.1
    a_pose.poses_body[1, 50] -= 0.1
    a_pose.poses_body[1, 38] += 0.2
    a_pose.poses_body[1, 41] -= 0.2

    # Wrist bend
    a_pose.poses_body[1, 62] -= 0.8
    a_pose.poses_body[1, 59] += 0.8

    a_pose.name = "push"

    a_pose.redraw()
    return a_pose

def pull():
    # Arm Back
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 37] += 0.2
    a_pose.poses_body[1, 40] -= 0.2
    a_pose.poses_body[1, 46] += 0.3
    a_pose.poses_body[1, 49] -= 0.3

    # Arm Flex
    a_pose.poses_body[1, 52] -= 0.9
    a_pose.poses_body[1, 55] += 0.9

    # Arm Down
    a_pose.poses_body[1, 47] -= 0.3
    a_pose.poses_body[1, 50] += 0.3
    a_pose.poses_body[1, 38] -= 0.2
    a_pose.poses_body[1, 41] += 0.2

    a_pose.name = "pull"

    a_pose.redraw()
    return a_pose

def lift():
    a_pose = Body.a_pose(frames=2)

    # Arm Up
    a_pose.poses_body[1, 47] = 0
    a_pose.poses_body[1, 50] = 0
    a_pose.poses_body[1, 45] -= 0.1
    a_pose.poses_body[1, 48] -= 0.1
    a_pose.poses_body[1, 38] = 0.3
    a_pose.poses_body[1, 41] = -0.3

    # Arm Forward
    a_pose.poses_body[1, 46] -= 0.3
    a_pose.poses_body[1, 49] += 0.3
    a_pose.poses_body[1, 37] -= 0.6
    a_pose.poses_body[1, 40] += 0.6

    # Elbow twist
    a_pose.poses_body[1, 51] -= 1.2
    a_pose.poses_body[1, 54] -= 1.2

    a_pose.name = "lift"

    a_pose.redraw()
    return a_pose

def slouch():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 6] += 0.44
    a_pose.poses_body[1, 15] += 0.3
    # Clavicle slightly forward
    # a_pose.poses_body[1, 40] += 0.1
    # a_pose.poses_body[1, 37] -= 0.1

    # Bend at waist
    a_pose.poses_root[1, 0] += 0.15

    # Correct legs
    a_pose.poses_body[1, 0] -= 0.15
    a_pose.poses_body[1, 3] -= 0.15

    # Head
    a_pose.poses_body[1, 33] -= 0.3

    # Arm Down
    a_pose.poses_body[0, 47] -= 0.2
    a_pose.poses_body[0, 50] += 0.2
    a_pose.poses_body[0, 38] -= 0.2
    a_pose.poses_body[0, 41] += 0.2

    # Arm Down
    a_pose.poses_body[:, 47] -= 0.1
    a_pose.poses_body[:, 50] += 0.1
    a_pose.poses_body[:, 38] -= 0.1
    a_pose.poses_body[:, 41] += 0.1

    # Arm Forward
    a_pose.poses_body[:, 46] -= 0.1
    a_pose.poses_body[:, 49] += 0.1

    a_pose.poses_body[1, 39] -= 0.3
    a_pose.poses_body[1, 36] -= 0.3

    a_pose.poses_body[1, 46] -= 0.5
    a_pose.poses_body[1, 49] += 0.5

    # Arm Flex
    a_pose.poses_body[0, 52] -= 0.1
    a_pose.poses_body[0, 55] += 0.1
    a_pose.poses_body[1, 52] -= 0.5
    a_pose.poses_body[1, 55] += 0.5

    a_pose.name = "slouch"

    a_pose.redraw()
    return a_pose

def arm_forward_and_flex():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 46] -= 0.25
    a_pose.poses_body[1, 49] += 0.25
    a_pose.poses_body[1, 37] -= 0.5
    a_pose.poses_body[1, 40] += 0.5

    a_pose.poses_body[1, 52] -= 0.3
    a_pose.poses_body[1, 55] += 0.3

    # Elbow twist
    a_pose.poses_body[1, 51] -= 1.2
    a_pose.poses_body[1, 54] -= 1.2


    a_pose.poses_body[1, 38] -= 0.1
    a_pose.poses_body[1, 41] += 0.1
    a_pose.redraw()
    return a_pose

def arm_forward():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 46] -= 0.25
    a_pose.poses_body[1, 49] += 0.25
    a_pose.poses_body[1, 37] -= 0.5
    a_pose.poses_body[1, 40] += 0.5
    a_pose.redraw()
    return a_pose

def arm_forward_alt():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 46] -= 0.2
    a_pose.poses_body[1, 49] += 0.2
    a_pose.poses_body[1, 37] -= 0.4
    a_pose.poses_body[1, 40] += 0.4
    a_pose.redraw()
    return a_pose


def arm_back():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 37] += 0.4
    a_pose.poses_body[1, 40] -= 0.4
    a_pose.poses_body[1, 46] += 0.2
    a_pose.poses_body[1, 49] -= 0.2
    a_pose.redraw()
    return a_pose


def arm_up():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 47] = 0
    a_pose.poses_body[1, 50] = 0
    a_pose.poses_body[1, 45] -= 0.5
    a_pose.poses_body[1, 48] -= 0.5
    a_pose.poses_body[1, 38] = 0.1
    a_pose.poses_body[1, 41] = -0.1
    a_pose.redraw()
    return a_pose


def arm_up_alt():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 47] = 0
    a_pose.poses_body[1, 50] = 0
    a_pose.poses_body[1, 45] -= 0.1
    a_pose.poses_body[1, 48] -= 0.1
    a_pose.poses_body[1, 38] = 0.1
    a_pose.poses_body[1, 41] = -0.1
    a_pose.redraw()
    return a_pose


def arm_down():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 47] -= 0.5
    a_pose.poses_body[1, 50] += 0.5
    a_pose.poses_body[1, 38] -= 0.2
    a_pose.poses_body[1, 41] += 0.2
    a_pose.redraw()
    return a_pose


def arm_flex():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 52] -= 0.8
    a_pose.poses_body[1, 55] += 0.8
    a_pose.redraw()
    return a_pose

def arm_ext():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 52] = 0.0
    a_pose.poses_body[1, 55] = 0.0
    a_pose.redraw()
    return a_pose

def torso_bend_forward():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 6] += 0.64
    a_pose.poses_body[1, 15] += 0.3
    # Clavicle slightly forward
    a_pose.poses_body[1, 40] += 0.1
    a_pose.poses_body[1, 37] -= 0.1
    # Head
    a_pose.poses_body[1, 33] -= 0.2

    # Arms down
    a_pose.poses_body[1, 47] -= 0.2
    a_pose.poses_body[1, 50] += 0.2
    a_pose.poses_body[1, 38] -= 0.1
    a_pose.poses_body[1, 41] += 0.1

    # Arm Flex
    a_pose.poses_body[1, 52] -= 0.4
    a_pose.poses_body[1, 55] += 0.4

    a_pose.redraw()
    return a_pose

def torso_bend_forward_alt():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 6] += 0.44
    a_pose.poses_body[1, 15] += 0.2
    # Clavicle slightly forward
    a_pose.poses_body[1, 40] += 0.1
    a_pose.poses_body[1, 37] -= 0.1
    # Head
    a_pose.poses_body[1, 33] -= 0.2

    # Arms down
    a_pose.poses_body[1, 47] -= 0.2
    a_pose.poses_body[1, 50] += 0.2
    a_pose.poses_body[1, 38] -= 0.1
    a_pose.poses_body[1, 41] += 0.1

    # Arm Flex
    a_pose.poses_body[1, 52] -= 0.4
    a_pose.poses_body[1, 55] += 0.4

    a_pose.redraw()
    return a_pose


def torso_bend_left():
    a_pose = Body.a_pose(frames=2)
    # Tilt spine
    a_pose.poses_body[1, 8] -= 0.35
    a_pose.poses_body[1, 17] -= 0.25
    # Tilt root
    a_pose.poses_root[1, 2] -= 0.15
    # Correct legs
    a_pose.poses_body[1, 2] += 0.15
    a_pose.poses_body[1, 5] += 0.15
    a_pose.redraw()
    return a_pose

def torso_bend_right():
    a_pose = Body.a_pose(frames=2)
    # Tilt spine
    a_pose.poses_body[1, 8] += 0.35
    a_pose.poses_body[1, 17] += 0.25
    # Tilt root
    a_pose.poses_root[1, 2] += 0.15
    # Correct legs
    a_pose.poses_body[1, 2] -= 0.15
    a_pose.poses_body[1, 5] -= 0.15
    a_pose.redraw()
    return a_pose


def torso_twist_left():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 7] += 0.35
    a_pose.poses_body[1, 16] += 0.35
    a_pose.poses_body[1, 25] += 0.2
    a_pose.redraw()
    return a_pose


def torso_twist_right():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 7] -= 0.35
    a_pose.poses_body[1, 16] -= 0.35
    a_pose.poses_body[1, 25] -= 0.2
    a_pose.redraw()
    return a_pose


def knee_flex():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 9] = 1.5
    a_pose.poses_body[1, 12] = 1.5
    a_pose.redraw()
    return a_pose

def knee_flex_left():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 9] = 1.5
    a_pose.redraw()
    return a_pose

def knee_flex_right():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 12] = 1.5
    a_pose.redraw()
    return a_pose

def hips_ext_left():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 0] -= 0.85
    a_pose.poses_body[1, 2] += 0.3
    a_pose.redraw()
    return a_pose

def hips_ext_right():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 3] -= 0.85
    a_pose.poses_body[1, 5] -= 0.3

    a_pose.redraw()
    return a_pose


def hips_flex_left():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 0] += 0.7
    a_pose.redraw()
    return a_pose


def hips_flex_right():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 3] += 0.7
    a_pose.redraw()
    return a_pose


def hips_abd_left():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 2] += 0.85
    a_pose.redraw()
    return a_pose

def hips_abd_right():
    a_pose = Body.a_pose(frames=2)
    a_pose.poses_body[1, 5] -= 0.85
    a_pose.redraw()
    return a_pose

def lunge_left():
    body = Body.from_amass(
        os.path.join(C.datasets.amass, "totalcapture/s1/rom3_poses.npz"),
        start_frame=5000,
        end_frame=7000,
        name="lunge",
        sub_frames=[0,100],
        load_betas=False,
        normalize_root=True)
    body.poses_body[:, 2] += 0.1
    body.poses_body[:, 5] -= 0.1
    return body

def lunge_right():
    body = Body.from_amass(
        os.path.join(C.datasets.amass, "totalcapture/s1/rom3_poses.npz"),
        start_frame=5000,
        end_frame=7000,
        name="lunge",
        sub_frames=[0,333],
        load_betas=False,
        normalize_root=True)
    body.poses_body[:, 2] += 0.1
    body.poses_body[:, 5] -= 0.1
    return body

def leg_raise_left():
    body = Body.from_amass(
        os.path.join(C.datasets.amass, "totalcapture/s1/rom3_poses.npz"),
        start_frame=3900,
        end_frame=4515,
        name="lunge",
        sub_frames=[0,1],
        load_betas=False,
        normalize_root=True)
    return body

def leg_raise_right():
    body = Body.from_amass(
        os.path.join(C.datasets.amass, "totalcapture/s1/rom3_poses.npz"),
        start_frame=3900,
        end_frame=4515,
        name="lunge",
        # sub_frames=[0,614],
        load_betas=False,
        normalize_root=True)
    return body


def crouch():
    body = Body.from_amass(
        os.path.join(C.datasets.amass, "MPI_mosh/00096/simple_crouch_poses.npz"),
        start_frame=0,
        end_frame=818,
        name="crouch",
        sub_frames=[200, 285, 285],
        load_betas=False,
        normalize_root=True)

    body.poses_body[:, 9] = 0.0
    body.poses_body[:, 12]= 0.0

    body.poses_body[1, 1] -= 0.5
    body.poses_body[1, 4] += 0.5

    body.poses_body[-1, 1] += 0.3
    body.poses_body[-1, 4] -= 0.3
    body.redraw()
    return body




def poses_lower_body():
    bodies = []
    bodies.append(hips_ext_left())
    # bodies.append(hips_abd_left())
    bodies.append(hips_flex_left())
    bodies.append(knee_flex_left())
    bodies.append(hips_ext_right())
    # bodies.append(hips_abd_right())
    bodies.append(hips_flex_right())
    bodies.append(knee_flex_right())

    all_poses_body, all_poses_root = [], []
    for i, b in enumerate(bodies):
        all_poses_body = to_numpy(b.poses_body) if i == 0 else np.concatenate(
            (all_poses_body, to_numpy(b.poses_body[[1]])))
    for i, b in enumerate(bodies):
        all_poses_root = to_numpy(b.poses_root) if i == 0 else np.concatenate(
            (all_poses_root, to_numpy(b.poses_root[[1]])))
    return Body(poses_body=all_poses_body, poses_root=all_poses_root, smpl_layer=STARLayer(device=C.device), name="poses_lower_body")


def poses_upper_body():
    bodies = []
    bodies.append(arm_forward())
    bodies.append(arm_up())
    # bodies.append(arm_flex())
    # bodies.append(arm_back())
    bodies.append(torso_bend_forward())

    all_poses_body, all_poses_root = [], []
    for i, b in enumerate(bodies):
        all_poses_body = to_numpy(b.poses_body) if i == 0 else np.concatenate(
            (all_poses_body, to_numpy(b.poses_body[[1]])))
    for i, b in enumerate(bodies):
        all_poses_root = to_numpy(b.poses_root) if i == 0 else np.concatenate(
            (all_poses_root, to_numpy(b.poses_root[[1]])))
    return Body(poses_body=all_poses_body, poses_root=all_poses_root, smpl_layer=STARLayer(device=C.device), name="poses_upper_body")


def from_pose(pose):
    return eval(pose)()
