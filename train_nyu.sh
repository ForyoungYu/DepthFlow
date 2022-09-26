#!/bin/bash

DATAPATH='C:\dataset\nyu\sync'
EVALPATH='C:\dataset\nyu\official_splits\test'

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
    # --random_crop_ratio 0.86 \
    # --resume ''
    # --lr 0.0001 \
    # --wd 1e-2 \
    # --distributed