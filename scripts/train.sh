#!/bin/bash
python main.py \
    --train \
    --total_epochs 4 \
    --cuda \
    --batch_size 1 \
    --path_to_dataset ../unicityscape \
    --path_to_checkpoints ../snapshots