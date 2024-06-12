DATASET=./data/TartanAir_shibuya
SAVEDIR=logs/shibuya

mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt

for SCENE in Standing01 Standing02 RoadCrossing03 RoadCrossing04 RoadCrossing05 RoadCrossing06 RoadCrossing07
do
    python main/eval.py \
    --config-path=../configs \
    --config-name=shibuya \
    data.imagedir=$DATASET/$SCENE/image_0 \
    data.gt_traj=$DATASET/$SCENE/gt_pose.txt \
    data.savedir=$SAVEDIR \
    data.calib=calibs/tartan_shibuya.txt \
    data.name=replica-$SCENE-$SEQ \
    save_video=false 
done


