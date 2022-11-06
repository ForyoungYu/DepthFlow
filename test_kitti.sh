#!/bin/bash

# Windows Path
# DATAPATH='C:\dataset\kitti\input'
# GTPATH='C:\dataset\kitti\gt_depth'
# Linux Path
DATAPATH='dataset/kitti/input'
GTPATH='dataset/kitti/gt_depth'
MAXDEPTH=150
MINDEPTH=1

python evaluate.py \
    --save_dir 'test_output' \
    --dataset 'kitti' \
    --data_path $DATAPATH \
    --gt_path  $DATAPATH \
    --data_path_eval $EVALPATH \
    --gt_path_eval $EVALPATH \
    --garg_crop \
    --checkpoint_path '' \
