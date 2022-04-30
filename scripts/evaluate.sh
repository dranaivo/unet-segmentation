#!/bin/bash
python ../segmentation/main.py \
    --test \
    --batch_size 1 \
    --cuda \
    --path_to_dataset ../unicityscape \
    --path_to_checkpoints ../snapshots