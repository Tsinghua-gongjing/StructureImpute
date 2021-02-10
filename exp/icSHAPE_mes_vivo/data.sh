#!/bin/bash
work_space=$(dirname $0)
python structureimpute/dataset/generate_data.py \
    --sequence_length 100 \
    --filename_train data/xk/icSHAPE_mes_vivo.train \
    --filename_validation data/xk/icSHAPE_mes_vivo.val \
