script_dir=/home/gongjing/project/shape_imputation/StructureImpute/scripts
work_space=$(pwd)

time CUDA_VISIBLE_DEVICES=$1 python $script_dir/main.py \
                                            --loaded_pt_file /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/windowLen100.sliding100.validation.prediction_trainHasNull_lossAll.pt \
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
                                            --filename_train /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/train_randomnullfragment_DA/DA50.txt \
                                            --filename_validation /home/gongjing/project/shape_imputation/data/hek_wc_vivo/3.shape/shape.c200T2M0m0.out.windowsHasNull/low_depth_null/sampling/windowLen100.sliding100.validation.low_50_1234.txt \
                                            --filename_prediction ./prediction.txt \
                                            --logfile $work_space/log.txt
