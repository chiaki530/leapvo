# DATASET=/storage/user/chwe/Datasets/MPI-Sintel-complete/training

# source /usr/wiss/chwe/miniconda3/bin/activate midas-py310

# for SCENE in alley_2 ambush_4 ambush_5 ambush_6 cave_2 cave_4 market_2 market_5 market_6 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3 
# do
#     SCENE_PATH=$DATASET/final/$SCENE
#     python image_folder_to_depth.py \
#     --img_dir=$SCENE_PATH \
#     --save_dir=$DATASET/monodepth/zoedepth/$SCENE \
#     --model=ZoeD_N 
# done



# DATASET=/storage/user/chwe/Datasets/TartanAir_shibuya

# source /usr/wiss/chwe/miniconda3/bin/activate midas-py310

# for SCENE in Standing01 Standing02 RoadCrossing03 RoadCrossing04 RoadCrossing05 RoadCrossing06 RoadCrossing07
# do
#     SCENE_PATH=$DATASET/$SCENE/image_0
#     python image_folder_to_depth.py \
#     --img_dir=$SCENE_PATH \
#     --save_dir=$DATASET/monodepth/zoedepth/$SCENE \
#     --model=ZoeD_N 
# done



DATASET=/storage/user/chwe/Datasets/rgbd_bonn_dataset

source /usr/wiss/chwe/miniconda3/bin/activate midas-py310

for SCENE in balloon balloon2 balloon_tracking balloon_tracking2 crowd crowd2 crowd3 moving_nonobstructing_box moving_nonobstructing_box2 person_tracking person_tracking2 
do
    SCENE_PATH=$DATASET/rgbd_bonn_${SCENE}/rgb
    python image_folder_to_depth.py \
    --img_dir=$SCENE_PATH \
    --save_dir=$DATASET/monodepth/zoedepth/$SCENE \
    --model=ZoeD_N 
done