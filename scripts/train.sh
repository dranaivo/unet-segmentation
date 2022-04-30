#!/bin/bash
python ../segmentation/main.py \
    --train \
    --total_epochs 8 \
    --cuda \
    --batch_size 1 \
    --resume \
    --save \
    --save_epoch 2 \
    --path_to_dataset ../unicityscape \
    --path_to_checkpoints ../snapshots