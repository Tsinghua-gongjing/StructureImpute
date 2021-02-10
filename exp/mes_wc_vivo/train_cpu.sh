#!/bin/bash
work_space=$(dirname $0)
outdir=$work_space/out
if [ ! -d $outdir ] ;then 
    mkdir $outdir
fi
# time CUDA_VISIBLE_DEVICES=$1 
python structureimpute/engine/main.py \
    --loaded_pt_file test/prediction.pt \
    --arch AllFusionNetMultiply \
    --batch_size 100 \
    --test_batch_size 100 \
    --epochs 4000 \
    --lr 0.0001 \
    --train_type trainHasNull_lossAll \
    --monitor_val_loss train_hasnull_validate_hasnull \
    --sequence_length 100 \
    --sliding_length 100 \
    --lstm_hidden_size 128 \
    --lstm_num_layers 2 \
    --lstm_bidirectional \
    --use_residual \
    --save_model \
    --no_cuda \
    --filename_train data/icSHAPE_mes_vivo/icSHAPE_mes_vivo.val \
    --filename_validation data/icSHAPE_mes_vivo/icSHAPE_mes_vivo.val \
    --filename_prediction $outdir/prediction.txt \
    --logfile $outdir/log.txt