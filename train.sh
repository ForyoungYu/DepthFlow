#!/bin/bash

python train.py \
    --name 'MyNet' \
    --dataset 'nyu' \
    --epochs 10 \
    --bs 2 \
    --workers 8 \
    --same_lr