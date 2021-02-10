#!/bin/bash
work_space=$(dirname $0)
python -u structureimpute/dataset/generate_data.py \
    --sequence_length 100 \
    --filename_train $1 \
