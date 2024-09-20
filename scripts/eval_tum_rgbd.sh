DATASET=/storage/group/dataset_mirrors/01_incoming/TUM_RGBD_Dataset
SAVEDIR=logs/tum_rgbd

mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt


# for NAME in balloon balloon2 balloon_tracking balloon_tracking2 removing_nonobstructing_box moving_obstructing_box \
# person_tracking person_tracking2 placing_nonobstructing_box2 placing_nonobstructing_box3 synchronous2 
# for NAME in freiburg3_walking_halfsphere freiburg3_walking_rpy freiburg3_walking_static freiburg3_walking_xyz
for NAME in freiburg3_sitting_halfsphere frieburg3_sitting_rpy frieburg3_sitting_xyz freiburg3_sitting_static
do
    SCENE=rgbd_dataset_${NAME}  
    python main/eval.py \
    --config-path=../configs \
    --config-name=bonn_rgbd \
    data.imagedir=$DATASET/$SCENE/rgb \
    data.gt_traj=$DATASET/$SCENE/groundtruth.txt \
    data.savedir=$SAVEDIR \
    data.calib=calibs/tum_rgbd_fr3.txt \
    data.name=tum_rgbd-$SCENE \
    save_video=true 
done


