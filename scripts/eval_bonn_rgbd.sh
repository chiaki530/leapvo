DATASET=/storage/slurm/chwe/datasets/rgbd_bonn_dataset
SAVEDIR=logs/bonn_rgbd

mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt


for NAME in balloon balloon2 balloon_tracking balloon_tracking2 removing_nonobstructing_box moving_obstructing_box \
person_tracking person_tracking2 placing_nonobstructing_box2 placing_nonobstructing_box3 synchronous2 

do
    SCENE=rgbd_bonn_${NAME}  
    python main/eval.py \
    --config-path=../configs \
    --config-name=bonn_rgbd \
    data.imagedir=$DATASET/$SCENE/rgb \
    data.gt_traj=$DATASET/$SCENE/cam_gt_aligned.txt \
    data.savedir=$SAVEDIR \
    data.calib=calibs/bonn_rgbd.txt \
    data.name=replica-$SCENE-$SEQ \
    save_video=true 
done


