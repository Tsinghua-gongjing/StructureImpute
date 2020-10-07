script_dir=/home/gongjing/project/shape_imputation/ShapeImputation/scripts
work_space=/home/gongjing/project/shape_imputation/exper/$4 # S4: model

predict(){
echo "predict: "$2", "$3
time CUDA_VISIBLE_DEVICES=$1 python $script_dir/main.py --load_model_and_predict  \
        --loaded_pt_file $work_space/prediction.pt \
        --arch AllFusionNetMultiply \
        --batch_size 100 \
        --test_batch_size 10000 \
        --epochs 4000 \
        --lr 0.0001 \
        --train_type trainHasNull_lossAll \
        --sequence_length 100 \
        --sliding_length 100 \
        --lstm_hidden_size 128 \
        --lstm_num_layers 2 \
        --lstm_bidirectional  \
        --use_residual  \
        --save_model  \
        --filename_train /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment/windowLen100.sliding100.train.randomNperfragmentNullPct0.3.maxL20.x10.txt \
        --filename_validation $2 \
        --filename_prediction $3 \
        --logfile $3.log
}

predict $1 $2 $3 # $1: GPU, $2: data, $3: savefn predict.txt