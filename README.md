# broader_sdoh_hf_readmission

This repo contains our code for the paper _Large Language Models for Integrating Social Determinant of Health Data: A Case Study on Heart Failure 30-Day Readmission Prediction_.

# Requirements
Heart failure 30-day hospital readmission prediction:
```
python 3.9
imblearn==0.0
joblib==1.2.0
numpy==1.24.4
pandas==2.0.0
pymongo==4.7.0
scikit_learn==1.4.2
shap==0.45.0
tqdm==4.65.0
xgboost==1.7.6
```


# Datasets
## Datasets

The social determinants of health (SDOH) datasets used in this study can be found below:

|   Dataset | Number of SDOH variables Used |
|---------------- | -------------- |
| [AHRQ SDOHD](https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html) |  760 |


# Reproducibility

## 1. Heart Failure (HF) Readmission Prediction
The patient dataset is unavailable due to privacy reasons --- however the following commands demonstrate the steps we used to train and evaluate binary classification models (using clinical and public SDOH data):

To train binary classification models on HF 30-day hospital readmission prediction (in file, choose classification algorithm, features):
```
python classification_driver_nestKfold.py
```
To analyze results of the HF models:
```
python analyze_classification_perf_results.py
```
To analyze fairness of the HF models:
```
python fairness_analyze_results.py
```
To get feature importance information (from trained XGBoost models):
```
python analyze_XGB_SHAP.py
```

## 2. Code to generate all tables and plots can be found in ```/data/```

## All Related Documents: 
- [Expanded SDOH variables used from AHRQ SDOHD](https://zenodo.org/records/14291734?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI1NTlhZTZhLTYwY2EtNDJkNS1iYTNhLTZjMGRhNWU2ZGNiYyIsImRhdGEiOnt9LCJyYW5kb20iOiI0MDdiMWU4Nzc4MmQzNzEwOWVmMTFlOTIzZjYwODI1ZiJ9.RXHy7WwMZR46nhAV989LX3zCdW_TyO-lxC7HRMVTWcp7kt9mWySZKmAVO9jTAJX10oOnF4ezbisPBE1Z13dAtw)