import rerun as rr

import numpy as np
import os

from pathlib import Path
from itertools import chain

import pdb

DATASET = '/storage/user/chwe/Datasets/MPI-Sintel-complete/training'
SCENE = 'alley_2'
DEPTH_IMAGE_SCALING = 1.0

def load_depth(depthdir):
    depth_exts = ["*.npy"]
    depth_list = sorted(chain.from_iterable(Path(depthdir).glob(e) for e in depth_exts))

    depth_maps = []
    for depth in depth_list:
        depth_maps.append(np.load(depth))

    
    depth_maps = np.stack(depth_maps)
    print(depth_maps.shape)
    return depth_maps

def visualize(depth_maps):
    pdb.set_trace()
    rr.init("test_zoedepth", spawn=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    
    S, H, W, _ = depth_maps.shape
    for s in range(S):
        rr.log(
            f"world/camera",
                rr.Pinhole(
                    resolution=[W, H],
                    focal_length=W,
                    camera_xyz=rr.ViewCoordinates.RDF,  # FIXME LUF -> RDF
            ),
        )

        rr.log(f"world/camera/image/depth", rr.DepthImage(depth_maps[s], meter=DEPTH_IMAGE_SCALING))
        rr.set_time_sequence("frame", s)

if __name__ == '__main__':
    depthdir = f"{DATASET}/monodepth/zoedepth/{SCENE}"
    depth_maps = load_depth(depthdir)
    visualize(depth_maps)

    