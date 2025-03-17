## 0. Environment 

pip install -r requirements.txt

## 1. Datasets download & Directory setting

Download the dev dataset from here: [**Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23)**](https://doi.org/10.5281/zenodo.7709052) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7709052.svg)](https://doi.org/10.5281/zenodo.7709052)

Unzip the data (mic_dev, metadata_dev, foa_dev) into a folder named "data_2024" and set the folder structure as shown below:

```
SR_BERG
├── cst_former
├── logs
└── data_2024
    ├── foa_dev
    ├── metadata_dev
    └── mic_dev
```


## 2. Feature Extraction

**NGCC + MS**:
```
python batch_feature_extraction.py 9
```
This result will be saved in the 'seld_feat_label' folder inside the 'data_2024' directory."
## 3. SELD Training


**Train CST-Former w/ NGCC+MS**
```
python3 train_seldnet.py 333 my_experiment
```
You can adjust rel-distance weight in seldnet_model.py and epoch, learning rate in parameters.py
baseline :
    - rel-distance weight = 1.5
    - nb_epoch = 50
    - learning rate = 1e-5

**Reproduce Experiment Result**
1. Change `nb_epochs` to 1 and `lr` to 0 in parameters.py
2. Change params['pretrained_model_weights'] to `models_audio/333_new_dist_dev_split0_multiaccdoa_mic_gcc_model.h5` in parameters.py (in argv==333)

and run the following command
```
python3 train_seldnet.py 333 check_best
```

| Model | F<sub>20°</sub> | DOAE<sub>CD</sub> | RDE<sub>CD</sub> | weights |
| ----| --- | --- | --- | --- |
| new_dist | 30.2% | 22.64&deg; | 0.34 | `models_audio/333_new_dist_dev_split0_multiaccdoa_mic_gcc_model.h5` |

**Check Latency**

If you want to check latency, run the following command

```
python3 train_seldnet.py 333 latency_check
```
| Model | latency (GPU) | weights |
| ---- | --- | --- |
| CST-Former Small w/ NGCC | 11ms (A100) | `models/333_cst-3t16c.h5` |
| CST-Former Large w/ NGCC | 11ms (A100) | `models/333_cst-3t16c-large.h5` |
