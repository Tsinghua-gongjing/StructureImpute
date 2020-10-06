# StructureImpute
Deep learning based imputation of RNA secondary structure profile.

## Overall architecture of StructureImpute model

![](/StructureImpute_framework.png)

## Installation

```
pip install -r requirements.txt
```


## Train a model

* 1. make a new directory and copy the `train.sh` script into the directory
* 2. modify the parameters to proper value, including `--script_dir`,`--filename_train`,`--filename_validation`
* 3. execute the script with specified GPU like `bash train.sh 2,3`

