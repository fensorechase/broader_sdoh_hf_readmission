# broader_sdoh_hf_readmission

This repo contains our code for the paper _Exploring Broader Integration of Social Determinants of Health for Predicting Readmission of Patients with Heart Failure_.

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
## 1. Prepare patient data, gather SDOH data and merge  ```/data/```:
Run ``` /data/patient_inclusion_Circ_2024.Rmd ``` to apply study inclusion, exclusion criteria.

Then run ``` /data/merge_SDOH_Circ_2024.Rmd ``` to merge SDOH data with patients.

## 2. Heart Failure (HF) Readmission Prediction
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

## 3. Code to generate all tables and plots can be found in ```/data/```
First, pull readmission prediction model results from your local MongoDB: 
- ```/scripts/analyze_classification_perf_results.py ``` to gather prediction performance values.
- ```/scripts/fairness_analyze_results.py ``` to gather prediction fairness values.
- ```/scripts/analyze_XGB_SHAP.py``` to gather feature importance values for XGBoost models.

Then, to generate the patient characteristics table, use ```table1_Circ_2024.Rmd```.

To tabulate model performance and fairness run ```calculate_HF_performance_2024.Rmd```, and to tabulate feature importance run ```plot_HF_SHAP_Circ_2024.Rmd```.

Note that all SDOH features used from AHRQ SDOHD can be found below in Related Documents.

## All Related Documents: 
- [Expanded SDOH variables used from AHRQ SDOHD](https://zenodo.org/records/14291734?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI1NTlhZTZhLTYwY2EtNDJkNS1iYTNhLTZjMGRhNWU2ZGNiYyIsImRhdGEiOnt9LCJyYW5kb20iOiI0MDdiMWU4Nzc4MmQzNzEwOWVmMTFlOTIzZjYwODI1ZiJ9.RXHy7WwMZR46nhAV989LX3zCdW_TyO-lxC7HRMVTWcp7kt9mWySZKmAVO9jTAJX10oOnF4ezbisPBE1Z13dAtw)