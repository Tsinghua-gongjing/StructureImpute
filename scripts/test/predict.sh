script_dir=/home/gongjing/project/shape_imputation/ShapeImputation/scripts
work_space=$(pwd)

predict(){
echo "predict: "$2", "$3
time CUDA_VISIBLE_DEVICES=$1 python $script_dir/main.py --load_model_and_predict \
        --loaded_pt_file $work_space/prediction.pt \
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
        --lstm_bidirectional  \
        --use_residual  \
        --save_model  \
        --filename_train /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment_DA/DA50.txt \
        --filename_validation $2 \
        --filename_prediction $work_space/prediction.$3.txt \
        --logfile $work_space/log.$3.txt
}

# validation_FXR_exceed=/home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt.sort.shape100.txt
# predict $1 $validation_FXR_exceed "FXR_exceed"
predict $1 $2 $3

# bash predict.sh 2 /home/gongjing/project/shape_imputation/data/CLIP/human_trx_clip/CLIPDB20123_FXR2_HEK293_trx.tx_has_shape_region_null_exceed.fimo/fimo.new.FXR.txt.sort.shape100.txt "FXR_exceed"