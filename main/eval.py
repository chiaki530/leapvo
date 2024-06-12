import math
import os

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from main.leapvo import LEAPVO
from main.stream import dataset_stream, sintel_stream, video_stream
from main.utils import (eval_metrics, load_traj, plot_trajectory,
                        save_trajectory_tum_format, update_timestamps)


@hydra.main(version_base=None, config_path="configs", config_name="demo")
def main(cfg: DictConfig):

    slam = None
    skip = 0

    imagedir, calib, stride, skip = (
        cfg.data.imagedir,
        cfg.data.calib,
        cfg.data.stride,
        cfg.data.skip,
    )

    if os.path.isdir(imagedir):
        if cfg.data.traj_format == "sintel":
            dataloader = sintel_stream(imagedir, calib, stride, skip)
        else:
            dataloader = dataset_stream(
                imagedir, calib, stride, skip, mode=cfg.data.traj_format
            )

    else:
        dataloader = video_stream(imagedir, calib, stride, skip)

    image_list = []
    intrinsics_list = []
    for i, (t, image, intrinsics) in enumerate(tqdm(dataloader)):
        if t < 0:
            break

        image_list.append(image)
        intrinsics_list.append(intrinsics)
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        # initialization
        if slam is None:
            slam = LEAPVO(cfg, ht=image.shape[1], wd=image.shape[2])

        slam(t, image, intrinsics)

    pred_traj = slam.terminate()

    if "gt_traj" in cfg.data and cfg.data.gt_traj != "":
        gt_traj = load_traj(
            cfg.data.gt_traj,
            cfg.data.traj_format,
            skip=cfg.data.skip,
            stride=cfg.data.stride,
        )
    else:
        gt_traj = None

    os.makedirs(f"{cfg.data.savedir}/{cfg.data.name}", exist_ok=True)

    pred_traj = list(pred_traj)
    if gt_traj is not None:
        if cfg.data.traj_format in ["tum", "tartanair"]:
            pred_traj[1] = update_timestamps(
                cfg.data.gt_traj, cfg.data.traj_format, cfg.data.skip, cfg.data.stride
            )

    if cfg.save_trajectory:
        save_trajectory_tum_format(
            pred_traj, f"{cfg.data.savedir}/{cfg.data.name}/leapvo_traj.txt"
        )

    if cfg.save_video:
        slam.visualizer.save_video(filename=cfg.slam.PATCH_GEN)

    if cfg.save_plot:
        plot_trajectory(
            pred_traj,
            gt_traj=gt_traj,
            title=f"LEAPVO Trajectory Prediction for {cfg.exp_name}",
            filename=f"{cfg.data.savedir}/{cfg.data.name}/traj_plot.pdf",
        )

    if gt_traj is not None:
        ate, rpe_trans, rpe_rot = eval_metrics(
            pred_traj,
            gt_traj=gt_traj,
            seq=cfg.exp_name,
            filename=os.path.join(cfg.data.savedir, cfg.data.name, "eval_metrics.txt"),
        )
        with open(os.path.join(cfg.data.savedir, "error_sum.txt"), "a+") as f:
            line = f"{cfg.data.name:<20} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
            f.write(line)
            line = f"{ate:.5f}\n{rpe_trans:.5f}\n{rpe_rot:.5f}\n"
            f.write(line)

    # # visualization
    # if cfg.viz:
    #     vis_rerun(slam, image_list, intrinsics_list)


if __name__ == "__main__":
    main()
