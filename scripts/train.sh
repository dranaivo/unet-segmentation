#!/bin/bash
python ../segmentation/main.py \
    --train \
    --total_epochs 4 \
    --path_to_dataset ../unicityscape \
    --path_to_checkpoints ../snapshots