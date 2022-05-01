script_dir=/root/StructureImpute

predict(){
echo "predict: "$2", "$3

python $script_dir/structureimpute/dataset/generate_data.py --filename_train $2 --sequence_length 100

time CUDA_VISIBLE_DEVICES=$1 python $script_dir/tools/main.py --predict \
        --load_model $4 \
        --filename_validation $2 \
        --filename_prediction $3
}


predict $1 $2 $3 $4 # $1: GPU, $2: data, $3: savefn predict.txt


# python3 $script_dir/tools/main.py --predict --load_model /root/StructureImpute/data/meta_model.pt --filename_validation /root/StructureImpute/data/test.txt --filename_prediction /root/StructureImpute/data/test.impute.txt

# python structureimpute/dataset/generate_data.py --filename_train /root/StructureImpute/data/GSE120724_1cell.icshape.out.h100.txt.predict/iteration1/allfragment.0.5.txt2 --sequence_length 100