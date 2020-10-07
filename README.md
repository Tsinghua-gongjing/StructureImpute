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
* 4. the model will be saved as `prediction.pt`

## Predict with the trained model

* 1. the prediction can be done by running `main.py` with parameter `--load_model_and_predict`,`--loaded_pt_file`,`--filename_validation` (the other paramters *must* same as `train.sh`)
* 2. the `test` directory includes a example `predict.sh`
