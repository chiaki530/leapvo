from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.tools import plot, file_interface

from matplotlib.collections import LineCollection

def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)

def load_timestamps(time_file, traj_format='replica'):
    if traj_format in ['tum', 'tartan_shibuya']:
        with open(time_file, 'r+') as f:
            lines = f.readlines()
        timestamps_mat = [float(x.split(' ')[0]) for x in lines if not x.startswith('#')]
        return timestamps_mat

def load_traj(gt_traj_file, traj_format='replica', skip=0, stride=1):
    gt_traj = None
    if gt_traj_file == '':
        return gt_traj
    
    if traj_format in ['replica']:
        gt_traj = load_replica_traj(gt_traj_file, skip=skip, stride=stride)
        return gt_traj
    elif traj_format in ['tum', 'tartan_shibuya', 'sintel', 'kitti_tum']:
        traj = file_interface.read_tum_trajectory_file(gt_traj_file)
        xyz = traj.positions_xyz
        # shift -1 column -> w in back column
        quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)

        # timestamps_mat = np.arange(xyz.shape[0]).astype(float)
        timestamps_mat = traj.timestamps
        traj_tum = np.column_stack((xyz, quat))
        return (traj_tum, timestamps_mat)
    
def eval_metrics(pred_traj, gt_traj=None, seq="", filename=""):
    pred_traj = make_traj(pred_traj)
    
    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)

        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(pred_traj.timestamps.shape[0],gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)
    
    # ATE
    traj_ref = gt_traj
    traj_est = pred_traj
    
    ate_result = main_ape.ape(traj_ref, traj_est, est_name='traj',
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    ate = ate_result.stats['rmse']

    # RPE rotation and translation
    delta_list = [1]
    rpe_rots, rpe_transs = [], []
    for delta in delta_list:
        rpe_rots_result = main_rpe.rpe(traj_ref, traj_est, est_name='traj',
            pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=True,
            delta=delta, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True)

        rot = rpe_rots_result.stats['rmse']
        rpe_rots.append(rot)

    for delta in delta_list:
        rpe_transs_result = main_rpe.rpe(traj_ref, traj_est, est_name='traj',
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True,
            delta=delta, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True)

        trans = rpe_transs_result.stats['rmse']
        rpe_transs.append(trans)
    
    rpe_trans, rpe_rot = np.mean(rpe_transs), np.mean(rpe_rots)
    with open(filename, 'w+') as f:
        f.write(f"Seq: {seq} \n\n")
        f.write(f"{ate_result}")
        f.write(f"{rpe_rots_result}")
        f.write(f"{rpe_transs_result}")
        
    print(f"Save results to {filename}")
    return ate, rpe_trans, rpe_rot

def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print("WARNING", pred_traj.timestamps.shape[0],gt_traj.timestamps.shape[0])
        
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: ' '.join(map(str, a))
    with Path(filename).open('w') as f:
        for i in range(traj.num_poses):
            f.write(f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[1,2,3,0]])}\n")
    print(f"Saved {filename}")


def load_gt_traj(gt_file, traj_format='kitti', skip=0, stride=1):


    traj_w_c = np.loadtxt(gt_file)
    traj_w_c = traj_w_c[skip::stride]
    assert traj_w_c.shape[1] == 12 or traj_w_c.shape[1] == 16
    
    poses = [np.array([[r[0], r[1], r[2], r[3]],
                    [r[4], r[5], r[6], r[7]],
                    [r[8], r[9], r[10], r[11]],
                    [0, 0, 0, 1]]) for r in traj_w_c]
    
    pose_path = PosePath3D(poses_se3=poses)
    timestamps_mat = np.arange(traj_w_c.shape[0]).astype(float)
    
    traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps_mat)
    xyz = traj.positions_xyz
    # shift -1 column -> w in back column
    quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    
    traj_tum = np.column_stack((xyz, quat))
    return (traj_tum, timestamps_mat)

def load_replica_traj(gt_file, format='kitti', skip=0, stride=1):
    traj_w_c = np.loadtxt(gt_file)
    traj_w_c = traj_w_c[skip::stride]
    assert traj_w_c.shape[1] == 12 or traj_w_c.shape[1] == 16
    
    poses = [np.array([[r[0], r[1], r[2], r[3]],
                       [r[4], r[5], r[6], r[7]],
                       [r[8], r[9], r[10], r[11]],
                       [0, 0, 0, 1]]) for r in traj_w_c]
    
    pose_path = PosePath3D(poses_se3=poses)
    timestamps_mat = np.arange(traj_w_c.shape[0]).astype(float)
    
    traj = PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps_mat)
    xyz = traj.positions_xyz
    # shift -1 column -> w in back column
    quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    
    traj_tum = np.column_stack((xyz, quat))
    return (traj_tum, timestamps_mat)


    
def save_pips_plot(rgbs, trajs, trajs_gt, save_name, save_dir, start_idx, valid_label=None, valid_gt=None):
    S, N = trajs.shape[:2]
    H, W = rgbs.shape[-2:]
    stride = 1
    margin = 100
    
    num_row = (S-1)//4 + 1
    num_col = min(S, 4)
    fig, axis = plt.subplots((S-1)//4 + 1, 4, figsize=(2 * num_col, 1.5 * num_row))
    
    # mask = gt in boundaries & trajs is valid
    mask = (np.ones((S,N)) > 0)
    if valid_label is not None:
        mask = mask & valid_label
    if valid_gt is not None:
        mask = mask & valid_gt
    #  padding = 0
    #  mask = (trajs_gt[...,0] >= padding) & (trajs_gt[...,0] < W - padding) & (trajs_gt[...,1] >= padding) & (trajs_gt[...,1] < H - padding) 

    for s in range(S):
        row = s // 4
        col = s % 4
        if S <= 4:
            ax = axis[col]
        else:
            ax = axis[row, col]
        
        ate = np.linalg.norm(trajs[s] - trajs_gt[s], axis=1).mean()
        masked_ate = np.linalg.norm(trajs[s][mask[s]] - trajs_gt[s][mask[s]], axis=1).mean()
        
        ax.set_title(f'T={start_idx + s}, masked_ate={masked_ate:.3f}, valid={valid_label[s,::stride].sum()}', fontsize=5, pad=0)
        ax.imshow(rgbs[s].transpose(1,2,0))
        # axis[s].scatter(trajs[s,::stride,0], trajs[s,::stride,1], s=1)
        if valid_label is not None:
            pts_vis = trajs[s,::stride][valid_label[s,::stride]]
            pts_occ = trajs[s,::stride][~valid_label[s,::stride]]
            ax.scatter(trajs_gt[s,::stride,0], trajs_gt[s,::stride,1], s=0.5, color='y', marker='^')
            ax.scatter(pts_vis[...,0], pts_vis[...,1], s=0.5, color='g', alpha=0.5)
            ax.scatter(pts_occ[...,0], pts_occ[...,1], s=0.5, color='r', alpha=0.5)

            # plot lines
            segs = [[(x[0], x[1]), (y[0], y[1])] for x, y in zip(trajs[s,::stride][mask[s,::stride]], trajs_gt[s,::stride][mask[s,::stride]])]
            line_segments = LineCollection(segs,
                               linewidths=0.5,
                               linestyles='-')
            ax.add_collection(line_segments)
        else:
            ax.scatter(trajs[s,::stride,0], trajs[s,::stride,1], s=1, color='g')
        ax.set_axis_off()
        ax.set_xlim(0-margin, W+margin)
        ax.set_ylim(H+margin, 0-margin)    # flip, otherwise would be upside down
        
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{save_name}.png', dpi=300)
    print(f"save to {save_dir}/{save_name}")
    plt.close()
