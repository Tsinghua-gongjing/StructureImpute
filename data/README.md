## Data process pipeline


### Download raw data 

The 

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

* icSHAPE_hek293_vivo: HEK293 cell line of *in vivo* whole cell condition
* icSHAPE_hek293_vitro: HEK293 cell line of *in vitro* whole cell condition
* icSHAPE_hek293_ch_vivo: HEK293 cell line of *in vivo* chromatin-associated component
* icSHAPE_hek293_cy_vivo: HEK293 cell line of *in vivo* cytoplasmic component
* icSHAPE_hek293_np_vivo: HEK293 cell line of *in vivo* nucleoplasmic component

DMS-seq data:

* DMSseq_K562_vivo: K562 cell line of *in vivo* whole cell
* DMSseq_K562_vitro: K562 cell line of *in vitro* whole cell
* DMSseq_fibroblast_vivo: fibroblast cell line of *in vivo* whole cell
* DMSseq_fibroblast_vivo: fibroblast cell line of *in vitro* whole cell


### Data augmentation
```
# generate NULL pattern file of shape.txt

python data_shape_distribution.py --data data/raw/structure_score/icSHAPE_hek293_vivo.train
# data/raw/structure_score/icSHAPE_hek293_vivo.train.null_pattern.txt will be created.


# cmd for data augmentation
python generate_train_DA.py --txt data/raw/structure_score/icSHAPE_hek293_vivo.train.null_pattern.txt \
	--strategy shadow_null_shuffle \
	--times 50
  
# data/raw/structure_score/icSHAPE_hek293_vivo_DA50.train will be created.

# Generate NPZ files
python structureimpute/dataset/generate_data.py \
    --filename_train data/raw/structure_score/icSHAPE_hek293_vivo_DA50.train \
    --sequence_length 100 \
# data/raw/structure_score/icSHAPE_hek293_vivo_DA50.train.npz will be created.


mv data/raw/structure_score/*.npz data/processed/

```

