#!/bin/bash

python train.py \
    --dataset 'kitti' \
    --data_path 'dataset/kitti/' \
    --gt_path 'dataset/kitti' \
    --filenames_file './train_test_inputs/kitti_eigen_train_files_with_gt.txt'
    --max_depth  \
    --min_depth  \
    --data_path_eval 'dataset/kitti' \
    --gt_path_eval 'dataset/kitti' \
    --filenames_file_eval 'train_test_inputs/kitti_eigen_train_files_with_gt.txt'
    --min_depth_eval \
    --max_depth_eval \