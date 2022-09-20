#!/bin/bash

python train.py \
    --name 'MyNet' \
    --dataset 'nyu' \
    --bs 10 \
    --workers 12 \
    --same_lr \
    --resume 'checkpoints/MyNet_19-Sep_16-45-nodebs10-tep25-lr0.000357-wd0.1-d99d94d7-bbf5-4ac9-a5ef-209eae0d4325_latest.pt'
    # --wd 1e-2 \
    # --epochs 10 \
    # --distributed