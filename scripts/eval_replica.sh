DATASET=./data/Replica_Dataset
SAVEDIR=logs/replica

mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt

for SCENE in  office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 
do
    for SEQ in Sequence_1
    do
        python main/eval.py \
        --config-path=../configs \
        --config-name=replica \
        data.imagedir=$DATASET/$SCENE/$SEQ/rgb \
        data.gt_traj=$DATASET/$SCENE/$SEQ/traj_w_c.txt \
        data.savedir=$SAVEDIR \
        data.calib=calibs/replica.txt \
        data.name=replica-$SCENE-$SEQ \
        save_video=false 
    done
done

