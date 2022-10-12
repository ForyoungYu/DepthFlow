#!/bin/bash

# Windows Path
DATAPATH='C:\dataset\nyu\sync'
EVALPATH='C:\dataset\nyu\official_splits\test'
# Linux Path
# DATAPATH='dataset/nyu/sync'
# EVALPATH='dataset/nyu/official_splits/test'

python evaluate.py \
    --save_dir 'test_output' \
    --dataset 'nyu' \
    --data_path $DATAPATH \
    --gt_path  $DATAPATH \
    --data_path_eval $EVALPATH \
    --gt_path_eval $EVALPATH \
    --garg_crop \
    --checkpoint_path '' \
