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
    --batch_size 100 \
    --test_batch_size 720 \
    --epochs 4000 \
    --lr 0.001 \
    --log_interval 100 \
    --train_type trainHasNull_lossAll \
    --monitor_val_loss train_hasnull_validate_onlynull \
    --sequence_length 100 \
    --early_stopping_num 40 \
    --sliding_length 100 \
    --lstm_hidden_size 128 \
    --lstm_num_layers 2 \
    --lstm_bidirectional \
    --use_residual \
    --save_model \
    --filename_train data/xk/icSHAPE_mes_vivo.train \
    --filename_validation data/xk/icSHAPE_mes_vivo.val \
    --filename_prediction $outdir/prediction.txt \
    --logfile $outdir/log.txt \
    |tee $outdir/log.txt

srun -p Test $work_space/eval.sh 
