#!/bin/bash
work_space=$(dirname $0)
outdir=$work_space/out
if [ ! -d $outdir ] ;then 
    mkdir $outdir
fi
# time CUDA_VISIBLE_DEVICES=$1 
name=$(basename $work_space)
python -u tools/main.py \
    --finetune \
    --load_model data/pretrained.pt \
    --batch_size 800 \
    --test_batch_size 720 \
    --lr 0.00001 \
    --log_interval 100 \
    --train_type DMSloss_all \
    --monitor_val_loss DMSloss_maskonly \
    --filename_train data/processed/${name}.train \
    --filename_validation data/processed/${name}.val \
    --filename_prediction $outdir/prediction.txt \
    --logdir $outdir/tfb \
    |tee -a $outdir/log.txt
