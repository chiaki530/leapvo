
DATASET=/storage/user/chwe/Datasets/rgbd_bonn_dataset
SAVEDIR=logs/mpi_sintel_depth_init/bonn_rgbd/cotracker_kernel_v2_cauchy_delta_anchor_50k_kf2_S16_init8_p64_s0.1_backward
mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# rgbd_bonn_balloon
for SCENE in rgbd_bonn_balloon
do
    python leapvo/cotrackerslam_depth_init.py \
            --config-path=../configs \
            --config-name=cotrackerslam_cotracker_kernel_v2_cauchy_delta \
            data.imagedir=$DATASET/$SCENE/rgb \
            data.gt_traj=$DATASET/$SCENE/cam_gt_aligned.txt \
            +data.depthdir=$DATASET/monodepth/zoedepth/$SCENE \
            data.savedir=$SAVEDIR \
            data.calib=calibs/tartan_shibuya.txt \
            data.name=replica-$SCENE-$SEQ \
            data.traj_format=kitti_tum \
            data.stride=1 \
            slam.PATCH_GEN=grid_grad_8 \
            slam.PATCHES_PER_FRAME=64 \
            save_video=false \
            save_results=false \
            plot=true \
            save_trajectory=true \
            slam.kf_stride=2 \
            slam.S_slam=12 \
            slam.num_init=8 \
            slam.backward_tracking=true \
            visualizer.tracks_leave_trace=8 \
            slam.VIS_THRESHOLD=0.9 \
            slam.CONF_THRESHOLD=1.0 \
            slam.CONF_QUANTILE=0.8 \
            slam.STATIC_QUANTILE=0.0 \
            slam.STATIC_THRESHOLD=0.1 \
    # evo_traj tum saved_trajectories/replica-$SCENE-$SEQ.txt --save_as_kitti

done



# DATASET=/storage/local/chwe/datasets/TartanAir_shibuya
# SAVEDIR=pips_slam/outputs/cotrackerslam/tartan_shibuya/vis_uncertainty_delta
# mkdir -p $SAVEDIR
# echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# # 00_1  01_0  01_1  02_0  02_1  02_2  03_0  04_0  07_1  08_0  09_0  10_0  10_1
# for SCENE in  Standing02 RoadCrossing03 RoadCrossing04 RoadCrossing05 RoadCrossing06 RoadCrossing07
# do

#     python pips_slam/cotrackerslam.py \
#             --config-path=configs \
#             --config-name=cotrackerslam_cotracker_kernel_v2_cauchy_delta \
#             data.imagedir=$DATASET/$SCENE/image_0 \
#             data.gt_traj=$DATASET/$SCENE/gt_pose.txt \
#             data.savedir=$SAVEDIR \
#             data.calib=pips_slam/calibs/tartan_shibuya.txt \
#             data.name=replica-$SCENE-$SEQ \
#             data.traj_format=kitti_tum \
#             data.stride=1 \
#             slam.PATCH_GEN=grid_grad_8 \
#             slam.PATCHES_PER_FRAME=64 \
#             save_video=true \
#             plot=true \
#             save_trajectory=true \
#             slam.kf_stride=4 \
#             slam.S_slam=8 \
#             slam.num_init=8 \
#             slam.PATCH_LIFETIME=12 \
#             slam.STATIC_QUANTILE=0.0 \
#             slam.STATIC_THRESHOLD=0.1 \
#             slam.CONF_THRESHOLD=1.0 \
#             slam.CONF_QUANTILE=0.8 \
#             slam.backward_tracking=true \
#             visualizer.tracks_leave_trace=8 \
#     # evo_traj tum saved_trajectories/replica-$SCENE-$SEQ.txt --save_as_kitti

# done