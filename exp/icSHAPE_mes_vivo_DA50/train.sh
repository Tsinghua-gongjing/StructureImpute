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
    --lr 0.001 \
    --train_type trainHasNull_lossAll \
    --monitor_val_loss train_hasnull_validate_onlynull \
    --filename_train data/processed/${name}.train \
    --filename_validation data/processed/${name}.val \
    --filename_prediction $outdir/prediction.txt \
    --logdir $outdir/tfb \
    |tee $outdir/log.txt

srun -p Test $work_space/eval.sh 
