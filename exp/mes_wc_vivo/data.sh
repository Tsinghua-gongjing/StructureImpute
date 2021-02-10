#!/bin/bash
work_space=$(dirname $0)
python structureimpute/dataset/generate_data.py \
    --sequence_length 100 \
    --filename_train data/icSHAPE_mes_vivo/icSHAPE_mes_vivo.train \
    --filename_validation data/icSHAPE_mes_vivo/icSHAPE_mes_vivo.val \