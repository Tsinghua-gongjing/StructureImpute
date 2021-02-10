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
    -J test -K \
    python -u structureimpute/engine/main.py \
    --load_model_and_continue_train \
    --loaded_pt_file data/xk/prediction.pt \
    --arch AllFusionNetMultiply \
    --batch_size 800 \
    --test_batch_size 720 \
    --lr 0.001 \
    --train_type trainHasNull_lossAll \
    --monitor_val_loss train_hasnull_validate_onlynull \
    --lstm_bidirectional \
    --use_residual \
    --save_model \
    --filename_train data/xk/icSHAPE_mes_vivo_DA50.train \
    --filename_validation data/xk/icSHAPE_mes_vivo.val \
    --filename_prediction $outdir/prediction.txt \
    --logfile $outdir/log.txt \
    |tee $outdir/log.txt

srun -p Test $work_space/eval.sh 
