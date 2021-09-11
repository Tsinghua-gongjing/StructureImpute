# StructureImpute
Deep learning based imputation of RNA secondary structure profile.

##  Architecture of StructureImpute model

![](misc/StructureImpute_framework.png)


## Installation


### Requirements

* torch>=1.1.0
* tensorboardX>=1.8
* nested_dict>=1.61
* pyfasta>=0.5.2
* matplotlib>=2.2.2
* seaborn>=0.9.0
* scikit-learn>=0.21.2

First clone the repository

```
git clone https://github.com/Tsinghua-gongjing/StructureImpute.git
```

Then install the required packages:

```
cd StructureImpute
pip install -r requirements.txt
pip install -e .
```

## Datasets

You can also downloaded all the data sets from [figshare](https://doi.org/10.6084/m9.figshare.16606850) or another cloud source:

```
# 3.1GB
wget -c https://cloud.tsinghua.edu.cn/f/c4a768fca5f64b02b1c2/?dl=1 -O data.tar.gz
# MD5 (data.tar.gz) = e532d847e92e6426f512eb4039272043 


cd StructureImpute
tar zxvf data.tar.gz
```

The data used in the study are under `data` directory, including:

* structure\_score: structural score of training/validation data set across different conditions
* CLIPDB: structural score around FXR2 binding sites
* RBMbase: structural score around m6A modification sites
* start\_stop\_codon: structural score around start/stop condon 

The processed training and validation structural core data set can be found under `data/structure_score` directory, including:

icSHAPE data:

* hek\_wc\_vivo: HEK293 cell line of *in vivo* whole cell condition
* hek\_wc\_vitro: HEK293 cell line of *in vitro* whole cell condition
* hek\_ch\_vivo: HEK293 cell line of *in vivo* chromatin-associated component
* hek\_cy\_vivo: HEK293 cell line of *in vivo* cytoplasmic component
* hek\_np\_vivo: HEK293 cell line of *in vivo* nucleoplasmic component

DMS-seq data:

* DMSseq\_K562\_vivo: K562 cell line of *in vivo* whole cell
* DMSseq\_K562\_vitro: K562 cell line of *in vitro* whole cell
* DMSseq\_fibroblast\_vivo: fibroblast cell line of *in vivo* whole cell
* DMSseq\_fibroblast\_vivo: fibroblast cell line of *in vitro* whole cell


Data augmentation
```
# generate NULL pattern file of shape.txt
# a /path/to/train.null_pattern.txt will be created
python data_shape_distribution.py --data /path/to/train.txt

# cmd for data augmentation
python generate_train_DA.py --txt /path/to/train.null_pattern.txt \
	--strategy shadow_null_shuffle \
	--times 50
```

### Generate NPZ files

```
python structureimpute/dataset/generate_data.py \
    --filename_train data/icSHAPE_mes_vivo/icSHAPE_mes_vivo.train \
    --sequence_length 100 \
```

You can directly download the pre-processed *.npz files with command lines below:

```
cd StructureImpute

# 3.9GB
wget -c https://cloud.tsinghua.edu.cn/f/a2f50d3cf63044339600/?dl=1 -O data_processed_icSHAPE.tgz
MD5 (data_processed_icSHAPE.tgz) = 076c52aa48cbe3b5260297225907a5be

# 3.6GB
wget -c https://cloud.tsinghua.edu.cn/f/32dc9c895d6748479976/?dl=1 -O data_processed_DMSseq.tgz
# MD5 (data_processed_DMSseq.tgz) = 87133f4c51bc968a62710987b8f4df3a

tar zxvf data_processed_icSHAPE.tgz data_processed_DMSseq.tgz
```

## Instructions

### Train a model

To train a model from scratch, run

```
exp/icSHAPE_hek293_ch/train_from_scratch.sh
```

To tune the training parameter by modifing the options `exp/icSHAPE_hek293_ch/train.sh`, show all the options by: 

```
python tools/main.py -h
```

Specific multi-GPU by:

```
CUDA_VISIBLE_DEVICES=0,1 exp/icSHAPE_hek293_ch/train.sh
```

Output files will be saved under the directory

* `log.txt`: the log file recording training process
* `model.pt`: the final saved model
* `prediction.txt`: the prediction result of validation set with the saved model

### Impute with the trained model

1. the prediction for a fragment file (e.g., data/structure\_score/hek\_wc\_vivo/validation.txt) can be done by running `tools/main.py` with parameter `--predict` and `--filename_prediction`

```
python tools/main.py \
    --predict \
    --load_model /path/to/output/dir/prediction.pt \
    --filename_validation /path/to/fragment/file/will/be/imputed.txt \
    --filename_prediction /path/to/fragment/file/for/saved/imputed.txt \
```

2. for a new icshape.out file, use the script `structureimpute/explore/predict_new_icshape.py`. A new directory `/path/to/new/icshape.out.predict` will be created and save all the bootstrapping results:

```
python structureimpute/explore/predict_new_icshape.py --icshape /path/to/new/icshape.out \
	--predict_model /path/to/output/dir/prediction.pt
```

### Fine-tune on DMSseq data

Using the `--finetune` option to fine-tune on specific data with pre-trained model (`data/pretrained.pt`).
```
exp/DMSseq_K562_vivo/train.sh
```



## Copyright and License

This project is free to use for non-commercial purposes - see the [LICENSE](https://github.com/Tsinghua-gongjing/StructureImpute/blob/master/LICENSE) file for details.

## Reference

Gong, Jing; Xu, Kui; Zhang, Qiangfeng Cliff (2021): A deep learning method for recovering missing signals in transcriptome-wide RNA structure profiles from probing experiments. figshare. Dataset. https://doi.org/10.6084/m9.figshare.16606850

