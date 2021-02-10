#!/bin/bash
work_space=$(dirname $0)
name=$(basename $work_space)
python structureimpute/explore/plot_one_method_in_multi_validation_null_corr_2.py \
    --validate_ls data/xk/icSHAPE_hek293_ch.val,data/xk/icSHAPE_hek293_ch.val \
    --predict_ls  $work_space/out/prediction.txt,$work_space/out/prediction.txt \
    --savefn  $work_space/out/prediction.pdf \
    --color_ls "#2BADE4,#9A0099" \
    --label_ls $name,$name
