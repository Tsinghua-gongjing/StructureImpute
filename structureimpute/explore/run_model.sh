expr=/home/gongjing/project/shape_imputation/exper

# cd $expr/b31_trainLossall_GmultiplyX_null0.1x10_nullfragmentL10x10
# bash train.sh 8,9,10,11

# cd $expr/b28_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10_randomNperValidate
# bash train.sh 8,9,10,11

# cd $expr/b17_trainLossall_GmultiplyX_nullfragmentL5x10
# bash train.sh 8,9,10,11

# cd $expr/b18_trainLossall_GmultiplyX_nullfragmentL10x10
# bash train.sh 8,9,10,11

# cd $expr/b19_trainLossall_GmultiplyX_nullfragmentL15x10
# bash train.sh 8,9,10,11

# cd $expr/b20_trainLossall_GmultiplyX_nullfragmentL20x10
# bash train.sh 8,9,10,11

# cd $expr/b22_trainLossall_GmultiplyX_nullfragmentL10x10_alltimeloss
# bash train.sh 8,9,10,11

# cd $expr/b25_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20
# bash train.sh 8,9,10,11

# cd $expr/b26_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20x10
# bash train.sh 8,9,10,11

# cd $expr/b27_trainLossall_GmultiplyX_randomNperfragmentpct0.3L20_randomNperValidate
# bash train.sh 8,9,10,11

for i in "${@:2}"
do
echo "process: "$i", with GPU: "$1
cd $expr/$i
bash train.sh $1
done