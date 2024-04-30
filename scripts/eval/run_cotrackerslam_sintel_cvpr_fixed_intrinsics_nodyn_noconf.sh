
DATASET=/storage/user/chwe/Datasets/MPI-Sintel-complete/training
SAVEDIR=logs/mpi_sintel/cotracker_kernel_v2_cauchy_delta_kf2_S12_init8_s0_conf1.0_backward_p100_nodyn_noconf
mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3
for SCENE in cave_2
do
    SCENE_PATH=$DATASET/final/$SCENE
    python leapvo/cotrackerslam.py \
    --config-path=../configs \
    --config-name=cotrackerslam_cotracker_kernel_v2_cauchy_delta \
    data.imagedir=$SCENE_PATH \
    data.gt_traj=$DATASET/cam_traj/$SCENE/traj_w_c_tum.txt \
    data.savedir=$SAVEDIR \
    data.calib=$DATASET/camdata_left/$SCENE \
    data.name=sintel-$SCENE \
    data.traj_format=sintel \
    data.stride=1 \
    slam.PATCH_GEN=grid_grad_10 \
    slam.PATCHES_PER_FRAME=100 \
    save_video=false \
    plot=true \
    save_trajectory=true \
    slam.kf_stride=2 \
    slam.S_slam=12 \
    slam.num_init=8 \
    slam.backward_tracking=true \
    visualizer.tracks_leave_trace=8 \
    slam.VIS_THRESHOLD=0.9 \
    slam.CONF_THRESHOLD=1.0 \
    slam.CONF_QUANTILE=1.0 \
    slam.STATIC_QUANTILE=0.0 \
    slam.STATIC_THRESHOLD=0 \
    save_results=false \
    # --config-name=pipsmultislam_new \
done


# # SAVEDIR=pips_slam/outputs/cotrackerslam/sintel_ablation/nodyn_1114_new_cotracker_kernel_v2_cauchy_50k_kf2_S12_init12_s0.1_conf.8_backward_confperframe
# SAVEDIR=pips_slam/outputs/cotrackerslam/sintel_ablation/vis_uncertainty_delta
# mkdir -p $SAVEDIR
# echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt
# # alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3
# for SCENE in alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3
# do
#     SCENE_PATH=$DATASET/final/$SCENE
#     python pips_slam/cotrackerslam.py \
#     --config-path=configs \
#     --config-name=cotrackerslam_cotracker_kernel_v2_cauchy_delta \
#     data.imagedir=$SCENE_PATH \
#     data.gt_traj=$DATASET/cam_traj/$SCENE/traj_w_c_tum.txt \
#     data.savedir=$SAVEDIR \
#     data.calib=pips_slam/calibs/sintel.txt \
#     data.name=sintel-$SCENE \
#     data.traj_format=sintel \
#     data.stride=1 \
#     slam.PATCH_GEN=grid_grad_10 \
#     slam.PATCHES_PER_FRAME=100 \
#     save_video=true \
#     plot=true \
#     save_trajectory=true \
#     slam.kf_stride=4 \
#     slam.S_slam=8 \
#     slam.num_init=8 \
#     slam.backward_tracking=true \
#     visualizer.tracks_leave_trace=8 \
#     slam.VIS_THRESHOLD=0.0 \
#     # slam.STATIC_QUANTILE=0.0 \
#     # slam.STATIC_THRESHOLD=0.1 \
#     # slam.CONF_THRESHOLD=1.0 \
#     # slam.CONF_QUANTILE=0.8 \
#     # slam.STATIC_QUANTILE=0.0 \
#     # slam.STATIC_THRESHOLD=0.1 \
#     # --config-name=pipsmultislam_new \
# done



# DATASET=/media/chwe/T7/datasets/Replica_Dataset
# SAVEDIR=pips_slam/outputs/cotrackerslam/replica/cotracker_w8
# mkdir -p $SAVEDIR
# echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt

# for SCENE in office_0 
# do
#     for SEQ in Sequence_1
#     do
#         python pips_slam/cotrackerslam.py \
#             --config-path=configs \
#             --config-name=cotrackerslam_cotracker_w8 \
#             data.imagedir=$DATASET/$SCENE/$SEQ/rgb \
#             data.gt_traj=$DATASET/$SCENE/$SEQ/traj_w_c.txt \
#             data.savedir=$SAVEDIR \
#             data.calib=pips_slam/calibs/replica.txt \
#             data.name=replica-$SCENE-$SEQ \
#             data.stride=1 \
#             data.max_length=50 \
#             plot=True \
#             save_trajectory=True \
#             save_video=True \
#             slam.kf_stride=2 \
#             slam.S_slam=8 \
#         # evo_traj tum saved_trajectories/replica-$SCENE-$SEQ.txt --save_as_kitti
#     done
# done