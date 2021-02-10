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
    python -u tools/main.py \
    --load_model_and_continue_train \
    --loaded_pt_file data/xk/prediction.pt \
    --arch AllFusionNetMultiply \
    --batch_size 800 \
    --test_batch_size 720 \
    --lr 0.00001 \
    --train_type DMSloss_all \
    --monitor_val_loss DMSloss_maskonly \
    --lstm_bidirectional \
    --use_residual \
    --save_model \
    --filename_train data/xk/${name}.train \
    --filename_validation data/xk/${name}.val \
    --filename_prediction $outdir/prediction.txt \
    --logfile $outdir/log.txt \
    |tee -a $outdir/log.txt

srun -p Test $work_space/eval.sh |tee -a $outdir/log.txt 
