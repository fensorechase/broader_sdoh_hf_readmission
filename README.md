# Beyond Composite Indices: Comprehensive Social Determinants Improve Heart Failure Readmission Prediction

This repo contains the code for the article _Beyond Composite Indices: Comprehensive Social Determinants Improve Heart Failure Readmission Prediction_.

## Requirements

Heart failure 30-day hospital readmission prediction:

```python
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

## Datasets

The social determinants of health (SDOH) datasets used in this study can be found below:

|   Dataset | Number of SDOH variables Used |
|---------------- | -------------- |
| [AHRQ SDOHD](https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html) |  760 |

## Reproducibility

### 1. Prepare patient data, gather SDOH data and merge  ```/data/```

Run 
```
/data/patient_inclusion.Rmd
```
to apply study inclusion, exclusion criteria.

Then run 
```
/data/merge_SDOH.Rmd
``` 
to merge SDOH data with patients.

### 2. Heart Failure (HF) Readmission Prediction

The patient dataset is unavailable due to privacy reasons --- however the following commands demonstrate the steps we used to train and evaluate binary classification models (using clinical and public SDOH data):

To train binary classification models on HF 30-day hospital readmission prediction (in file, choose classification algorithm, features):

```bash
python classification_driver_nestKfold.py
```

To analyze results of the HF models:

```bash
python analyze_classification_perf_results.py
```

To analyze fairness of the HF models:

```bash
python fairness_analyze_results.py
```

To get feature importance information (from trained XGBoost models):

```bash
python analyze_XGB_SHAP.py
```

### 3. Code to generate all tables and plots

First, pull readmission prediction model results from your local MongoDB collections:

- To gather prediction performance values:

```bash
    python /scripts/analyze_classification_perf_results.py
```

- To gather prediction fairness values:

```bash
    python /scripts/fairness_analyze_results.py 
```

- To gather feature importance values for XGBoost models:

```bash
    python /scripts/analyze_XGB_SHAP.py
```

- Then, to generate the patient characteristics table, use:

```bash
    /data/table1_generate.Rmd
 ```

- To tabulate model performance and fairness run:

```bash
    /data/calculate_HF_performance.Rmd
```

- To tabulate feature importance run:

```bash
    /data/plot_HF_SHAP.Rmd
```

Note that all SDOH features used from AHRQ SDOHD can be found below in Related Documents.

### All Related Documents:

- [Expanded SDOH variables used from AHRQ SDOHD](https://zenodo.org/records/14291734?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI1NTlhZTZhLTYwY2EtNDJkNS1iYTNhLTZjMGRhNWU2ZGNiYyIsImRhdGEiOnt9LCJyYW5kb20iOiI0MDdiMWU4Nzc4MmQzNzEwOWVmMTFlOTIzZjYwODI1ZiJ9.RXHy7WwMZR46nhAV989LX3zCdW_TyO-lxC7HRMVTWcp7kt9mWySZKmAVO9jTAJX10oOnF4ezbisPBE1Z13dAtw)

## Rates of missingness for all expanded SDOH variables (i.e., from AHRQ SDOHD) can be found in ```summary_statistics/```

- ```TotalCohort_missing_rates_by_race.csv```
- ```TotalCohort_missing_rates_by_readmission.csv```
- ```TotalCohort_missing_rates_by_readmission_black.csv```
- ```TotalCohort_missing_rates_by_readmission_white.csv```

### Folder Structure

```plaintext
data/
|-- adi-download-2020-tract/
|-- sdi-download-2019-tract/
|-- feat_base.json
|-- feat_column.json
|-- subgroup_cols_fast.json
|-- 2010-18-all-granularities-AHRQ-dict.xlsx
|-- count_features.py
|-- num_unique_SDOH_features.py
|-- calculate_HF_performance.Rmd
|-- merge_SDOH.Rmd
|-- patient_inclusion.Rmd
|-- plot_HF_SHAP.Rmd
|-- table1_generate.Rmd
|-- data_cleaners.R
|-- gen_chars.R

scripts/
|-- analyze_classification_perf_results.py
|-- analyze_XGB_SHAP.py
|-- classification_driver_nestKfold.py
|-- evalHelper.py
|-- fairness_analyze_results.py
|-- fake_patient_data.csv

summary_statistics/
|-- AHRQ_used_county_metadata.csv
|-- AHRQ_used_tract_metadata.csv
|-- missing_rates_final_allstates_modelinput.csv
|-- TotalCohort_missing_rates_by_race.csv
|-- TotalCohort_missing_rates_by_readmission_black.csv
|-- TotalCohort_missing_rates_by_readmission_white.csv
|-- TotalCohort_missing_rates_by_readmission.csv
|-- domains_lists.json
