#!/bin/bash

# Windows Path
# DATAPATH='C:\dataset\kitti\input'
# GTPATH='C:\dataset\kitti\gt_depth'
# Linux Path
DATAPATH='dataset/kitti/input'
GTPATH='dataset/kitti/gt_depth'
MAXDEPTH=150
MINDEPTH=1

python train.py \
    --name 'EFT_l3_kitti' \
    --dataset 'kitti' \
    --do_kb_crop \
    --same_lr \
    --epochs 100 \
    --validate_every 500 \
    --bs 4 \
    --workers 12 \
    --filenames_file './train_test_inputs\kitti_eigen_train_files_with_gt.txt' \
    --filenames_file_eval './train_test_inputs\kitti_eigen_test_files_with_gt.txt' \
    --data_path $DATAPATH \
    --gt_path $GTPATH \
    --data_path_eval $DATAPATH \
    --gt_path_eval $GTPATH \
    --max_depth $MAXDEPTH \
    --min_depth $MINDEPTH \
    --max_depth_eval $MAXDEPTH \
    --min_depth_eval $MINDEPTH \
    --input_height 320 \
    --input_width 1056 \
    # --resume '' \
