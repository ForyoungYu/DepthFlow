#!/bin/bash

# Windows Path
DATAPATH='C:\dataset\nyu\sync'
EVALPATH='C:\dataset\nyu\official_splits\test'
# Linux Path
# DATAPATH='dataset/nyu/sync'
# EVALPATH='dataset/nyu/official_splits/test'

python train.py \
    --name 'EFT_l3_nyu' \
    --dataset 'nyu' \
    --epochs 100 \
    --same_lr \
    --bs 5 \
    --workers 16 \
    --data_path $DATAPATH \
    --gt_path $DATAPATH \
    --data_path_eval $EVALPATH \
    --gt_path_eval $EVALPATH \
    # --resume ''
    # --lr 0.0001 \
    # --wd 1e-2 \
    # --distributed
    