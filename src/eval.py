import os
import math

from src.leapvo import LEAPVO
from src.stream import sintel_stream, dataset_stream
from src.rerun_visualizer import vis_rerun
from src.plot_utils import plot_trajectory, save_trajectory_tum_format, eval_metrics, load_traj, load_timestamps

import torch
import hydra
from omegaconf import DictConfig

import pdb

@hydra.main(version_base=None, config_path="configs", config_name="leapvo_sintel")
def main(cfg: DictConfig):

    gt_traj = load_traj(cfg.data.gt_traj, cfg.data.traj_format, skip=cfg.data.skip, stride=cfg.data.stride)

    slam = None
    skip = 0

    imagedir, calib, stride, skip = cfg.data.imagedir, cfg.data.calib, cfg.data.stride, cfg.data.skip 

    if cfg.data.traj_format == 'sintel':
        dataloader = sintel_stream(None, imagedir, calib, stride, skip)
    else:
        dataloader = dataset_stream(None, imagedir, calib, stride, skip, mode=cfg.data.traj_format)

    image_list = []
    intrinsics_list = []
    for i, (t, image, intrinsics) in enumerate(dataloader):

        if "max_length" in cfg.data and i >= cfg.data.max_length: break
        if t < 0: break
        
        image_list.append(image)
        intrinsics_list.append(intrinsics)
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()
         
        # initialization
        if slam is None:
            slam = LEAPVO(cfg, ht=image.shape[1], wd=image.shape[2])

        slam(t, image, intrinsics, depth_g=None, cam_g=None)

    pred_traj = slam.terminate()

    if cfg.data.traj_format == 'tum':
        traj_t_map_file = cfg.data.gt_traj.replace('groundtruth.txt', 'rgb.txt')
        pred_traj = list(pred_traj)
        pred_traj[1] = load_timestamps(traj_t_map_file, cfg.data.traj_format)
        pred_traj[1] = pred_traj[1][:pred_traj[0].shape[0]]
    elif cfg.data.traj_format == 'tartan_shibuya':
        traj_t_map_file = cfg.data.gt_traj.replace('gt_pose.txt', 'times.txt')
        pred_traj = list(pred_traj)
        pred_traj[1] = load_timestamps(traj_t_map_file, cfg.data.traj_format)
        pred_traj[1] = pred_traj[1][:pred_traj[0].shape[0]]
    

    os.makedirs(f"{cfg.data.savedir}/{cfg.data.name}", exist_ok=True)

    if cfg.save_results:
        save_results_path  = f"{cfg.data.savedir}/{cfg.data.name}/saved_results.npz"
        slam.save_results(save_results_path, imagedir=cfg.data.imagedir)

    if cfg.save_trajectory:
        save_trajectory_tum_format(pred_traj, f"{cfg.data.savedir}/{cfg.data.name}/leapvo_traj.txt")

    if cfg.plot:
        plot_trajectory(pred_traj, gt_traj=gt_traj, title=f"LEAPVO Trajectory Prediction for {cfg.exp_name}", filename=f"{cfg.data.savedir}/{cfg.data.name}/traj_plot.pdf")
    
    if cfg.save_video:
        slam.visualizer.save_video(filename=cfg.slam.PATCH_GEN)

    ate, rpe_trans, rpe_rot = eval_metrics(pred_traj, gt_traj=gt_traj, seq=cfg.exp_name, filename=os.path.join(cfg.data.savedir,cfg.data.name, 'eval_metrics.txt'))
    with open(os.path.join(cfg.data.savedir, 'error_sum.txt'), 'a+') as f:
        line = f"{cfg.data.name:<20} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
        f.write(line)
        line = f"{ate:.5f}\n{rpe_trans:.5f}\n{rpe_rot:.5f}\n"
        f.write(line)


    # visualization
    if cfg.viz:
        vis_rerun(slam, image_list, intrinsics_list)



if __name__ == '__main__':
    main()
