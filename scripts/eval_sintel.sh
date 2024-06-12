DATASET=./data/MPI-Sintel-complete/training
SAVEDIR=logs/sintel

mkdir -p $SAVEDIR
echo $(date "+%Y-%m-%d %H:%M:%S") >> $SAVEDIR/error_sum.txt

for SCENE in alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3
do
    SCENE_PATH=$DATASET/final/$SCENE
    python main/eval.py \
    --config-path=../configs \
    --config-name=sintel \
    data.imagedir=$SCENE_PATH \
    data.gt_traj=$DATASET/camdata_left/$SCENE \
    data.savedir=$SAVEDIR \
    data.calib=$DATASET/camdata_left/$SCENE \
    data.name=sintel-$SCENE \
    save_video=false  
done

