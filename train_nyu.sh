#!/bin/bash

python train.py \
    --name 'EFT_l3' \
    --dataset 'nyu' \
    --epochs 50 \
    --same_lr \
    --bs 12 \
    --workers 24 \
    --resume 'checkpoints/EFT_22-Sep_14-52-nodebs12-tep50-lr0.000357-wd0.1-f3651bc2-1d6e-4e75-b214-59eef4eafcac_latest.pt'
    # --lr 0.0001 \
    # --wd 1e-2 \
    # --distributed