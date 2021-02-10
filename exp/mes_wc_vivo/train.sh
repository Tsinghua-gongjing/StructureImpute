#!/bin/bash
work_space=$(dirname $0)
outdir=$work_space/out
if [ ! -d $outdir ] ;then 
    mkdir $outdir
fi
# time CUDA_VISIBLE_DEVICES=$1 
part=$1
n=$2
name=$(basename $work_space)
srun --mpi=pmi2 --gres=gpu:${n} \
    -p $part -n1 \
    --ntasks-per-node=1 \
    -J $name -K \
    python -u structureimpute/engine/main.py \
    --loaded_pt_file test/prediction.pt \
    --arch AllFusionNetMultiply \
    --batch_size 100 \
    --test_batch_size 100 \
    --lr 0.0001 \
    --train_type trainHasNull_lossAll \
    --monitor_val_loss train_hasnull_validate_hasnull \
    --lstm_bidirectional \
    --use_residual \
    --save_model \
    --filename_train data/icSHAPE_mes_vivo/icSHAPE_mes_vivo.train \
    --filename_validation data/icSHAPE_mes_vivo/icSHAPE_mes_vivo.val \
    --filename_prediction $outdir/prediction.txt \
    --logfile $outdir/log.txt
