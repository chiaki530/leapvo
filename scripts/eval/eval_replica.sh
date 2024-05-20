
# DATASET=/storage/local/chwe/datasets/MPI-Sintel-complete/training
# SAVEDIR=pips_slam/outputs/cotrackerslam/replica/cotracker_iid_kf2_S12_init8_backward
# mkdir -p $SAVEDIR
# echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# # alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3
# for SCENE in cave_4 
# do
#     SCENE_PATH=$DATASET/final/$SCENE
#     python pips_slam/cotrackerslam.py \
#     --config-path=configs \
#     --config-name=cotrackerslam_cotracker_iid \
#     data.imagedir=$SCENE_PATH \
#     data.gt_traj=$DATASET/cam_traj/$SCENE/traj_w_c_tum.txt \
#     data.savedir=$SAVEDIR \
#     data.calib=pips_slam/calibs/sintel.txt \
#     data.name=sintel-$SCENE \
#     data.traj_format=sintel \
#     data.stride=1 \
#     slam.PATCH_GEN=grid_grad_8 \
#     slam.PATCHES_PER_FRAME=64 \
#     save_video=true \
#     plot=true \
#     save_trajectory=true \
#     slam.kf_stride=2 \
#     slam.S_slam=12 \
#     slam.num_init=8 \
#     slam.backward_tracking=true \
#     visualizer.tracks_leave_trace=8 \
#     # --config-name=pipsmultislam_new \
# done



# DATASET=/storage/remote/atcremers24/chwe/datasets/Replica_Dataset
# SAVEDIR=pips_slam/outputs/cotrackerslam/replica_seq2/nodyn_cotracker_kernel_v2_cauchy_50k_kf1_S12_init8_backward
# mkdir -p $SAVEDIR
# echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# # office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 
# for SCENE in  office_1 office_2 office_3 office_4 room_0 room_1 room_2 
# do
#     for SEQ in Sequence_2
#     do
#         python pips_slam/cotrackerslam.py \
#                 --config-path=configs \
#                 --config-name=cotrackerslam_cotracker_kernel_v2_cauchy_no_dyn \
#                 data.imagedir=$DATASET/$SCENE/$SEQ/rgb \
#                 data.gt_traj=$DATASET/$SCENE/$SEQ/traj_w_c.txt \
#                 data.savedir=$SAVEDIR \
#                 data.calib=pips_slam/calibs/replica.txt \
#                 data.name=replica-$SCENE-$SEQ \
#                 data.stride=1 \
#                 slam.PATCH_GEN=grid_grad_8 \
#                 slam.PATCHES_PER_FRAME=64 \
#                 save_video=false \
#                 plot=true \
#                 save_trajectory=true \
#                 slam.kf_stride=1 \
#                 slam.S_slam=12 \
#                 slam.num_init=8 \
#                 slam.backward_tracking=true \
#                 visualizer.tracks_leave_trace=8 \
#         # evo_traj tum saved_trajectories/replica-$SCENE-$SEQ.txt --save_as_kitti
#     done
# done


DATASET=/storage/user/chwe/Datasets/Replica_Dataset
SAVEDIR=logs/replica/cotracker_kernel_v2_cauchy_delta_50k_kf2_S12_init8_backward_conf.8_p64_nodyn
mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 
for SCENE in  office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 
do
    for SEQ in Sequence_1
    do
        python leapvo/cotrackerslam.py \
                --config-path=../configs \
                --config-name=cotrackerslam_cotracker_kernel_v2_cauchy_delta_nodyn \
                data.imagedir=$DATASET/$SCENE/$SEQ/rgb \
                data.gt_traj=$DATASET/$SCENE/$SEQ/traj_w_c.txt \
                data.savedir=$SAVEDIR \
                data.calib=calibs/replica.txt \
                data.name=replica-$SCENE-$SEQ \
                data.stride=1 \
                slam.PATCH_GEN=grid_grad_8 \
                slam.PATCHES_PER_FRAME=64 \
                save_video=false \
                save_results=false \
                plot=true \
                save_trajectory=true \
                slam.kf_stride=2 \
                slam.S_slam=12 \
                slam.num_init=12 \
                slam.CONF_QUANTILE=0.8 \
                slam.backward_tracking=true \
                visualizer.tracks_leave_trace=8 \
        # evo_traj tum saved_trajectories/replica-$SCENE-$SEQ.txt --save_as_kitti
    done
done

# DATASET=/storage/remote/atcremers24/chwe/datasets/Replica_Dataset
# SAVEDIR=pips_slam/outputs/cotrackerslam/replica_seq1/vis_uncertainty_delta
# mkdir -p $SAVEDIR
# echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# # office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 
# for SCENE in office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 
# do
#     for SEQ in Sequence_1
#     do
#         python pips_slam/cotrackerslam.py \
#                 --config-path=configs \
#                 --config-name=cotrackerslam_cotracker_kernel_v2_cauchy_delta \
#                 data.imagedir=$DATASET/$SCENE/$SEQ/rgb \
#                 data.gt_traj=$DATASET/$SCENE/$SEQ/traj_w_c.txt \
#                 data.savedir=$SAVEDIR \
#                 data.calib=pips_slam/calibs/replica.txt \
#                 data.name=replica-$SCENE-$SEQ \
#                 data.stride=1 \
#                 slam.PATCH_GEN=grid_grad_8 \
#                 slam.PATCHES_PER_FRAME=64 \
#                 save_video=true \
#                 plot=true \
#                 save_trajectory=true \
#                 slam.kf_stride=4 \
#                 slam.S_slam=8 \
#                 slam.num_init=8 \
#                 slam.PATCH_LIFETIME=12 \
#                 slam.STATIC_QUANTILE=0.0 \
#                 slam.STATIC_THRESHOLD=0.1 \
#                 slam.CONF_THRESHOLD=1.0 \
#                 slam.CONF_QUANTILE=0.8 \
#                 slam.backward_tracking=true \
#                 visualizer.tracks_leave_trace=8 \
#         # evo_traj tum saved_trajectories/replica-$SCENE-$SEQ.txt --save_as_kitti
#     done
# done