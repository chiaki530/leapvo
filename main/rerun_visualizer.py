import numpy as np
import rerun as rr


def vis_rerun(slam, image_list, intrinsics_list):
    poses, intrinsics, pts, clrs, pts_valid, patches, tstamps = slam.get_results()
    S = tstamps.shape[0]

    H, W, _ = image_list[0].shape
    rr.init("cotrackerslam", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    for s in range(S):

        rr.set_time_sequence("frame", s)

        K = np.eye(3)
        K[0, 0] = intrinsics_list[s][0]
        K[1, 1] = intrinsics_list[s][1]
        K[0, 2] = intrinsics_list[s][2]
        K[1, 2] = intrinsics_list[s][3]

        img_rgb = image_list[s]
        cams_T_world = poses[s]
        rr.log(f"world/camera/image/rgb", rr.Image(img_rgb))

        rr.log(
            f"world/camera",
            rr.Transform3D(
                translation=cams_T_world[:3, 3], mat3x3=cams_T_world[:3, :3]
            ),
        )
        rr.log(
            f"world/camera",
            rr.Pinhole(
                resolution=[W, H],
                image_from_camera=K,
                camera_xyz=rr.ViewCoordinates.RDF,  # FIXME LUF -> RDF
            ),
        )

        valid_s = (pts_valid[: (s + 1)] > 0.5).reshape(-1)
        pts_s = pts[: (s + 1)].reshape(-1, 3)
        pts_s = pts_s[valid_s]

        mask = (pts_s[..., 2] > -10) & (pts_s[..., 2] < 10)
        pts_s = pts_s[mask]
        # colors = clrs[s]
        colors = [0, 255, 0]
        rr.log(f"world/points", rr.Points3D(pts_s, colors=colors))
