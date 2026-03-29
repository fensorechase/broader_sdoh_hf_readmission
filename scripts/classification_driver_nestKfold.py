"""
This file is a driver to run binary classification models for specified heart failure (HF) endpoint. Prints results (performance, fairness, feature importance/coefficients) to specificed local MongoDB collection.

"""

import os
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import json
import urllib.parse
from datetime import datetime

import imblearn
import numpy as np
import pandas as pd

# from sklearn.inspection import permutation_importance # For feature importance
import shap
import sklearn.ensemble as sken
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.tree as sktree
import tqdm
import xgboost as xgb
from evalHelper import (
    evaluate_results,
    evaluate_results_fairness,
    get_train_test,
    read_json,
)

# from focal_loss import BinaryFocalLoss
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from pymongo import MongoClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# For xgboost balanced weights
from sklearn.utils.class_weight import compute_sample_weight

MODEL_PARAMS = {
    "xgb": {
        "model": xgb.XGBClassifier(),
        "params": {
            "max_depth": [6],
            "n_estimators": [50, 500],
            "learning_rate": [0.01],
            "eval_metric": ["logloss"],
            "lambda": [1],
            "alpha": [0, 0.2],
        },
    },
    "rf": {
        "model": sken.RandomForestClassifier(),
        "params": {
            "max_depth": [6, 8],
            "min_samples_leaf": [5, 10],
            "n_estimators": [5, 10],
            "class_weight": ["balanced"],
        },
    },
    "logr": {  # lbfgs solver is used.
        "model": LogisticRegression(),
        "params": {
            "penalty": ["l2"],
            "max_iter": [2000],
            "solver": ["lbfgs"],
            "class_weight": ["balanced"],
        },
    },
}


"""
Returns a Python list of all values from a json object, discarding the keys.
"""


def json_extract_values(obj):
    if isinstance(obj, dict):
        values = []
        for key, value in obj.items():
            values.extend(json_extract_values(value))
        return values
    elif isinstance(obj, list):
        return obj
    else:
        return []


"""
Calculate bootstrapped confidence intervals for a metric
"""
def calculate_bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(len(y_true)), replace=True, n_samples=len(y_true))
        y_true_bootstrap = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        
        # Calculate the statistic
        bootstrap_stats.append(metric_func(y_true_bootstrap, y_pred_bootstrap))
    
    # Calculate lower and upper bounds for CI
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrap_stats, alpha * 100)
    upper_bound = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return lower_bound, upper_bound

"""
Calculate confidence intervals for all metrics
"""
def calculate_all_metric_cis(y_true, y_pred, y_pred_binary):
    ci_results = {}
    
    # For AUC
    ci_results['auc_ci'] = calculate_bootstrap_ci(
        y_true, y_pred, 
        lambda y, p: skm.roc_auc_score(y, p)
    )
    
    # For Average Precision Score (APS)
    ci_results['aps_ci'] = calculate_bootstrap_ci(
        y_true, y_pred, 
        lambda y, p: skm.average_precision_score(y, p)
    )
    
    # For Precision
    ci_results['precision_ci'] = calculate_bootstrap_ci(
        y_true, y_pred_binary, 
        lambda y, p: skm.precision_score(y, p)
    )
    
    # For Recall
    ci_results['recall_ci'] = calculate_bootstrap_ci(
        y_true, y_pred_binary, 
        lambda y, p: skm.recall_score(y, p)
    )
    
    # For F1
    ci_results['f1_ci'] = calculate_bootstrap_ci(
        y_true, y_pred_binary, 
        lambda y, p: skm.f1_score(y, p)
    )
    
    # For AUPRC
    ci_results['auprc_ci'] = calculate_bootstrap_ci(
        y_true, y_pred, 
        lambda y, p: skm.precision_recall_curve(y, p)[0].mean()
    )
    
    # For MCC
    ci_results['mcc_ci'] = calculate_bootstrap_ci(
        y_true, y_pred_binary, 
        lambda y, p: skm.matthews_corrcoef(y, p)
    )
    
    # For False Negative Rate (1 - TPR)
    ci_results['fnr_ci'] = calculate_bootstrap_ci(
        y_true, y_pred_binary, 
        lambda y, p: 1 - skm.recall_score(y, p)
    )
    
    # For True Negative Rate (TNR)
    ci_results['tnr_ci'] = calculate_bootstrap_ci(
        y_true, y_pred_binary, 
        lambda y, p: skm.confusion_matrix(y, p)[0, 0] / (skm.confusion_matrix(y, p)[0, 0] + skm.confusion_matrix(y, p)[0, 1])
    )
    
    # For False Positive Rate (FPR)
    ci_results['fpr_ci'] = calculate_bootstrap_ci(
        y_true, y_pred_binary, 
        lambda y, p: skm.confusion_matrix(y, p)[0, 1] / (skm.confusion_matrix(y, p)[0, 0] + skm.confusion_matrix(y, p)[0, 1])
    )
    
    return ci_results

"""
Calculate confidence intervals for fairness metrics
"""
def calculate_fairness_metric_cis(subgroup_preds_dict, n_bootstrap=1000, confidence=0.95):
    fairness_ci_results = {}
    
    # Extract test data and predictions for each subgroup
    subgroups = list(subgroup_preds_dict.keys())
    
    if len(subgroups) < 2:
        return {}  # Not enough subgroups for fairness metrics
    
    # Create bootstrap samples
    bootstrap_fairness_metrics = {
        'eo_ratio': [],
        'fpr_parity': [],
        'tpr_parity': [],
        'fnr_parity': [],
        'dp_ratio': []
    }
    
    successful_bootstraps = 0
    max_attempts = n_bootstrap * 3  # Allow more attempts to get valid bootstraps
    
    for attempt in range(max_attempts):
        if successful_bootstraps >= n_bootstrap:
            break
            
        # For each subgroup, resample test indices
        bootstrap_subgroup_preds = {}
        valid_bootstrap = True
        
        for g in subgroups:
            test_x, test_y, binary_preds = subgroup_preds_dict[g]
            
            # Skip if subgroup is too small for meaningful bootstrap
            if len(test_y) < 5:
                valid_bootstrap = False
                break
                
            indices = resample(range(len(test_y)), replace=True, n_samples=len(test_y))
            
            # Extract bootstrap samples
            bootstrap_test_y = test_y.iloc[indices] if hasattr(test_y, 'iloc') else test_y[indices]
            bootstrap_binary_preds = [binary_preds[i] for i in indices]
            
            # Check if bootstrap sample has both classes and some positive predictions
            unique_y = np.unique(bootstrap_test_y)
            unique_preds = np.unique(bootstrap_binary_preds)
            
            if len(unique_y) < 2 or len(unique_preds) < 2:
                valid_bootstrap = False
                break
                
            # Check if we have sufficient positive and negative cases
            n_positive = np.sum(bootstrap_test_y == 1)
            n_negative = np.sum(bootstrap_test_y == 0)
            pred_positive = np.sum(np.array(bootstrap_binary_preds) == 1)
            
            if n_positive < 2 or n_negative < 2 or pred_positive < 1:
                valid_bootstrap = False
                break
            
            bootstrap_subgroup_preds[g] = [None, bootstrap_test_y, bootstrap_binary_preds]
        
        if not valid_bootstrap:
            continue
            
        # Calculate fairness metrics on bootstrap samples
        try:
            eo_ratio, fpr_parity, tpr_parity, fnr_parity, dpr = evaluate_results_fairness(None, bootstrap_subgroup_preds)
            
            # Only add if all metrics are valid (not NaN)
            if all(not np.isnan(x) for x in [eo_ratio, fpr_parity, tpr_parity, fnr_parity, dpr]):
                bootstrap_fairness_metrics['eo_ratio'].append(eo_ratio)
                bootstrap_fairness_metrics['fpr_parity'].append(fpr_parity)
                bootstrap_fairness_metrics['tpr_parity'].append(tpr_parity)
                bootstrap_fairness_metrics['fnr_parity'].append(fnr_parity)
                bootstrap_fairness_metrics['dp_ratio'].append(dpr)
                successful_bootstraps += 1
                
        except (ZeroDivisionError, ValueError, RuntimeError):
            # Skip this bootstrap sample if calculation fails
            continue
    
    # Calculate CI for each fairness metric only if we have enough valid samples
    min_samples_for_ci = max(50, n_bootstrap // 20)  # At least 50 samples or 5% of requested
    alpha = (1 - confidence) / 2
    
    for metric in bootstrap_fairness_metrics:
        values = bootstrap_fairness_metrics[metric]
        
        if len(values) >= min_samples_for_ci:
            # Remove any remaining NaN values just in case
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) >= min_samples_for_ci:
                lower_bound = np.percentile(values, alpha * 100)
                upper_bound = np.percentile(values, (1 - alpha) * 100)
                fairness_ci_results[f'{metric}_ci'] = (lower_bound, upper_bound)
            else:
                # Not enough valid samples
                fairness_ci_results[f'{metric}_ci'] = (np.nan, np.nan)
        else:
            # Not enough valid samples
            fairness_ci_results[f'{metric}_ci'] = (np.nan, np.nan)
    
    return fairness_ci_results




# #########################################################################
def main():
    parser = argparse.ArgumentParser()

    # For local Mongo:
    # Your connection string may look slightly difference. You should copy it from your local MongoDB Compass cluster, after the cluster is started using "mongosh" in the terminal.
    parser.add_argument(
        "-mongo_url",
        default="mongodb://...", # <your_mongo_connection_string>
    )

    parser.add_argument("-mongo_db", default="<your_db_name>", help="database_name") # <your_db_name>
    parser.add_argument(
        "-mongo_col", default="<your_collection_name>", help="collection_type" # <your_collection_name>
    )
    # default information
    parser.add_argument(
        "-data_file",
        default="../data/<your_patient_cohort_data_with_features_merged.csv>", # <your_patient_cohort_data_with_features_merged.csv> ... Post-processed version of fake_patient_data.csv after (i.e., after it's been merged with approrpiate EHR and SDOH features).
        help="data file",
    )
    parser.add_argument(
        "-base_feat", default="../data/feat_base.json", help="base_features"
    )

    parser.add_argument(
        "-feat_file", default="../data/feat_column.json", help="model_features"
    )

    parser.add_argument(
        "-subgroup_file",
        default="../data/subgroup_cols_fast.json",
        help="subgroups_to_test",
    )
    parser.add_argument(
        "-endpoint", default="readmit30bin"
    )  # readmit30bin (binary readmission indicator for hospital readmission due to heart failure within 30 days), or your desired binary (0/1) endpoint column name from your input file.

    parser.add_argument(
        "--feats",
        nargs="+",
        default=[
            # Main text: Model predictive performance Across with differing predictor sets from Table 2.
            # NOTE 1: limited_EHR does include 7 demographic variables, to adjust all models. 
            # If limited_EHR is not included in the feature set, then these 7 demographic variables are manully included.
            # NOTE 2: About limited/routine vs. expanded/comprehensive -- Expanded SDOH data is a superset of limiited SDOH data.
            "limited_EHR", # [BASE MODEL]: Routine clinical only (base model) (11 predictors)
            "expanded_clinical", # Expanded clinical (39 predictors)
            "M6_total_ahrq_cty_DF1_nm", # Comprehensive SDOH only (752 predictors)
            "routine_clinical_standard_SDOH", # Routine clinical + standard SDOH (26 predictors)
            "routine_clinical_comprehensive_SDOH", # Routine clinical + comprehensive SDOH (778 predictors)
            "expanded_clinical_comprehensive_SDOH" # [FULL MODEL]: Expanded clinical and comprehensive SDOH (full model) (806 predictors)

            # Main text: SDOH domains and geographic levels
            "demo_DF1_nm_county_AHRQ_domain1",
            "demo_DF1_nm_county_AHRQ_domain2",
            "demo_DF1_nm_county_AHRQ_domain3",
            "demo_DF1_nm_county_AHRQ_domain4",
            "demo_DF1_nm_county_AHRQ_domain5",
            # Combination of all 5 county-level domains is included above ("M6_total_ahrq_cty_DF1_nm")
            "demo_DF1_nm_tract_AHRQ_domain1",
            "demo_DF1_nm_tract_AHRQ_domain2",
            "demo_DF1_nm_tract_AHRQ_domain3",
            "demo_DF1_nm_tract_AHRQ_domain4",
            "demo_DF1_nm_tract_AHRQ_domain5"
            "M6_total_ahrq_trct_DF1_nm" # Combination of all 5 tract-level domains.

            ##################
            # Additional results: evaluating intersection of comprehensive SDOH features at tract vs. county levels.
            "demo_nm_intersect_county_AHRQ", "demo_nm_intersect_tract_AHRQ",

        ],
    )

    args = parser.parse_args()

    # Setup mongo
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]
    fairness_mcol = mdb["<your_fairness_collection_name>"] # <your_fairness_collection_name>
    # raw_mcol = mdb["<your_raw_predictions_collection_name>"] # If you would like to store the raw binary model predictions for each patient (0/1).
    logcoeffs_mcol = mdb["<your_log_coeff_collection_name>"] # <your_log_coeff_collection_name> For logistic regression coefficients.
    # logCI_mcol = mdb["<your_log_ci_collection_name>"] # If you would like to store 95% CI of logistic regression coefficients.
    
    # New collection for 95% CI metrics
    ci_metrics_mcol = mdb["<your_ci_metrics_collection_name>"] # For confidence intervals of performance metrics (AUC, precision, recall, F1, AUPRC, MCC, FPR, TNR, FNR).
    ci_fairness_mcol = mdb["<your_ci_fairness_collection_name>"] # For confidence intervals of fairness metrics.
    # New collection for XGBoost SHAP values:
    shap_mcol = mdb["<your_shap_collection_name>"]  # To save SHAP values for for XGBoost

    df = pd.read_csv(args.data_file)
    base_feat = read_json(args.base_feat)
    feat_info = read_json(args.feat_file)
    subgroups_bins = read_json(args.subgroup_file)

    # Determine the feature sets
    feat_cols = {}
    for ft in args.feats:
        colset = set()
        # check if it's a base feature, if so update
        if ft in base_feat:
            colset.update(base_feat[ft])
        else:
            for ftbase in feat_info[ft]:
                colset.update(base_feat[ftbase])
        feat_cols[ft] = list(colset)

    # Determine subgroups within each subgroup bin:
    # subgroups = json_extract_values(subgroups_bins)

    # Use "for s in subgroup_keys" as outer loop to subgroups.
    # Then, access exact subgroup name: i.e., "for g in subgroup_bins[s]"
    subgroup_keys = subgroups_bins.keys()


    # Create a dictionary to store predictions for each encounter
    # Structure: {encounter_id: {model_feat_combo: prediction}}
    all_encounter_predictions = {}

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)




    # Begin main loop:
    for i in tqdm.tqdm(range(1, 11), desc="test-split"):
        train_df, test_df, train_y, test_y = get_train_test(df, i, label=args.endpoint)

        test_y = test_y.astype(int)  # Cast to ensure labels are integers

        # Reset test_df indices for subgroup indexing
        test_df = test_df.reset_index()

        for fname, fcolumns in tqdm.tqdm(feat_cols.items(), desc="feats", leave=False):
            base_res = {
                "file": args.data_file,
                "feat": fname,
                "endpoint": args.endpoint,
                "fold": i,
            }

            # for both train and test get only those columns
            train_x = train_df[fcolumns]

            # Apply imputation:
            imputer = SimpleImputer(
                missing_values=np.nan, strategy="median"
            )  # , keep_empty_features=True
            train_x = imputer.fit_transform(train_x)

            # Apply feature preprocessing: StandardScaler
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)

            for mname, mk_dict in tqdm.tqdm(
                MODEL_PARAMS.items(), desc="models", leave=False
            ):

                gs = skms.GridSearchCV(
                    mk_dict["model"],
                    mk_dict["params"],
                    cv=5,
                    n_jobs=4,
                    scoring="roc_auc",
                    refit=True,
                )

                if mname != "xgb":
                    gs.fit(train_x, train_y)
                else:
                    gs.fit(
                        train_x,
                        train_y,
                        sample_weight=compute_sample_weight("balanced", train_y),
                    )

                # If desired, you can save the *best* model file in a folder for later use:

                model = gs.best_estimator_
                bestmodel_train_params = gs.best_params_

                # Get current date and time
                current_time = datetime.now()
                # Format the date and time into a string
                unique_string = current_time.strftime("%Y%m%d%H%M%S%f")
                # Base filename
                base_filename = "./model_files/model"
                # Create a unique filename by appending the unique string
                bestmodel_unique_filename = f"{base_filename}_{unique_string}.joblib"

                # Note - uncomment this to save model into file: dump(model, bestmodel_unique_filename)

                # Loop through test eval for each subgroup
                for s in subgroup_keys:
                    # 1 entire dict is for 1 SET of subgroups
                    # (e.g., 1 dict for "race" subgroups, since "s" is "race" one iteration.)
                    subgroup_preds_dict = {}  # curr_subgroup: [test_x, sg_test_y]
                    # Loop through test eval for each subgroup

                    for g in subgroups_bins[s]:
                        # Get indices of rows that match curr_subgroup
                        cs_ind = test_df.loc[test_df[g] == 1].index
                        # For subgroups, select only current 'sg' from test_df & test_y.
                        sg_test_df = test_df[test_df[g] == 1]
                        sg_test_y = test_y.iloc[
                            cs_ind
                        ]  # use column indices for test_y, because no subgroup cols here

                        print("______________________________________")
                        print("FOLD: ", i, "MODEL: ", mname)

                        # Get only desired feature cols
                        test_x = sg_test_df[fcolumns]
                        # get the test encounter id
                        test_idx = sg_test_df["Encounter"]
                        auc, aps, y_hat, binary_predictions, precision, recall, f1, auc_precision_recall, fnr, tnr, fpr, mcc = evaluate_results(gs, test_x, sg_test_y, args.endpoint, imputer=imputer, scaler=scaler)

                        # Add predictions to encounter dictionary 
                        model_feat_combo = f"{mname}_{fname}"
                        for enc_idx, enc_id in enumerate(test_idx):
                            if enc_id not in all_encounter_predictions:
                                all_encounter_predictions[enc_id] = {}
                            all_encounter_predictions[enc_id][model_feat_combo] = binary_predictions[enc_idx]

                        # Calculate confidence intervals for metrics
                        ci_results = calculate_all_metric_cis(sg_test_y, y_hat, binary_predictions)
                        
                        # Track subgroups preds for fairness:
                        subgroup_preds_dict[g] = [test_x, sg_test_y, binary_predictions]

                        perf_res = {
                            "model": mname,
                            "ts": datetime.now(),
                            "auc": auc,
                            "aps": aps,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "auprc": auc_precision_recall,
                            "fnr": fnr,
                            "tnr": tnr, 
                            "fpr": fpr,
                            'mcc': mcc,
                            "sg_key": s,
                            "subgroup": g,
                            "test_samp_size": len(sg_test_y),
                            "bestmodel_train_params_dict": bestmodel_train_params,
                            "bestmodel_unique_filename": bestmodel_unique_filename
                        }
                        mcol.insert_one({**base_res, **perf_res})
                        
                        # Save confidence intervals for metrics
                        ci_metrics_res = {
                            "model": mname,
                            "ts": datetime.now(),
                            "sg_key": s,
                            "subgroup": g,
                            "test_samp_size": len(sg_test_y),
                            "auc_ci_lower": ci_results['auc_ci'][0],
                            "auc_ci_upper": ci_results['auc_ci'][1],
                            "aps_ci_lower": ci_results['aps_ci'][0],
                            "aps_ci_upper": ci_results['aps_ci'][1],
                            "precision_ci_lower": ci_results['precision_ci'][0],
                            "precision_ci_upper": ci_results['precision_ci'][1],
                            "recall_ci_lower": ci_results['recall_ci'][0],
                            "recall_ci_upper": ci_results['recall_ci'][1],
                            "f1_ci_lower": ci_results['f1_ci'][0],
                            "f1_ci_upper": ci_results['f1_ci'][1],
                            "auprc_ci_lower": ci_results['auprc_ci'][0],
                            "auprc_ci_upper": ci_results['auprc_ci'][1],
                            "mcc_ci_lower": ci_results['mcc_ci'][0],
                            "mcc_ci_upper": ci_results['mcc_ci'][1],
                            "fnr_ci_lower": ci_results['fnr_ci'][0],
                            "fnr_ci_upper": ci_results['fnr_ci'][1],
                            "tnr_ci_lower": ci_results['tnr_ci'][0],
                            "tnr_ci_upper": ci_results['tnr_ci'][1],
                            "fpr_ci_lower": ci_results['fpr_ci'][0],
                            "fpr_ci_upper": ci_results['fpr_ci'][1],
                            "bestmodel_train_params_dict": bestmodel_train_params,
                            "bestmodel_unique_filename": bestmodel_unique_filename
                        }
                        ci_metrics_mcol.insert_one({**base_res, **ci_metrics_res})

                        # Get SHAP for XGBoost
                        if mname == "xgb":
                            # Apply same pipeline preprocessing to test_x:
                            xgb_test_x = imputer.transform(test_x)
                            xgb_test_x = scaler.transform(xgb_test_x)

                            explainer = shap.Explainer(model)
                            shap_values = explainer(np.ascontiguousarray(xgb_test_x))
                            shap_importance = shap_values.abs.mean(0).values
                            sorted_idx = shap_importance.argsort()
                            ordered_shaps = shap_importance[sorted_idx]
                            names_ordered_shaps = np.array(fcolumns)[sorted_idx]
                            # Save Ordered shaps & names.
                            xg_shap_res = {
                                "model": mname,
                                "ts": datetime.now(),
                                "auc": auc,
                                "aps": aps,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "auprc": auc_precision_recall,
                                "fnr": fnr,
                                "tnr": tnr, 
                                "fpr": fpr,
                                'mcc': mcc,
                                "sg_key": s,
                                "subgroup": g,
                                "test_samp_size": len(sg_test_y),
                                "shap_ordered_names": names_ordered_shaps.tolist(),
                                "shap_ordered_importance": ordered_shaps.tolist(),
                                "bestmodel_train_params_dict": bestmodel_train_params,
                                "bestmodel_unique_filename": bestmodel_unique_filename
                            }
                            shap_mcol.insert_one({**base_res, **xg_shap_res})
                        
                        # Get logr coeff for LogisticRegression
                        if "logr" in mname:
                            log_coeffs = {
                                "model": mname,
                                "feat": fname,
                                "sg_key": s,
                                "subgroup": g,                                
                                "ts": datetime.now(),
                                "auc": auc,
                                "aps": aps,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "auprc": auc_precision_recall,
                                "fnr": fnr,
                                "tnr": tnr, 
                                "fpr": fpr,
                                "test_samp_size": len(sg_test_y),
                                "logr_feat_names": fcolumns,
                                "logr_coeffs": model.coef_.tolist(),
                                "logr_intercept": model.intercept_.tolist(),
                                "bestmodel_train_params_dict": bestmodel_train_params,
                                "bestmodel_unique_filename": bestmodel_unique_filename
                            }
                            # Save logr coefficients. For later comparison btw black, white, & other subgroups.
                            logcoeffs_mcol.insert_one({**base_res, **log_coeffs})
                    
                    # Save fairness for this combo of mname + fname.
                    # After subgroup loop, calculate fairness:
                    eo_ratio, fpr_parity, tpr_parity, fnr_parity, dpr = evaluate_results_fairness(gs, subgroup_preds_dict)
                    
                    # Calculate confidence intervals for fairness metrics
                    fairness_ci_results = calculate_fairness_metric_cis(subgroup_preds_dict)
                    
                    fair_res = {
                        "model": mname,
                        "ts": datetime.now(),
                        "eo_ratio": eo_ratio,
                        "fpr_parity": fpr_parity,
                        "tpr_parity": tpr_parity,
                        "fnr_parity": fnr_parity,
                        "dp_ratio": dpr,
                        "sg_key": s,
                        "prot_attributes": subgroups_bins[s],
                        "total_test_samp_size": len(test_y),
                        "bestmodel_train_params_dict": bestmodel_train_params,
                        "bestmodel_unique_filename": bestmodel_unique_filename
                    }
                    fairness_mcol.insert_one({**base_res, **fair_res})
                    
                    # Save confidence intervals for fairness metrics
                    if fairness_ci_results:  # Only save if we have valid CIs (need at least 2 subgroups)
                        fairness_ci_res = {
                            "model": mname,
                            "ts": datetime.now(),
                            "sg_key": s,
                            "prot_attributes": subgroups_bins[s],
                            "total_test_samp_size": len(test_y),
                            "eo_ratio_ci_lower": fairness_ci_results['eo_ratio_ci'][0],
                            "eo_ratio_ci_upper": fairness_ci_results['eo_ratio_ci'][1],
                            "fpr_parity_ci_lower": fairness_ci_results['fpr_parity_ci'][0],
                            "fpr_parity_ci_upper": fairness_ci_results['fpr_parity_ci'][1],
                            "tpr_parity_ci_lower": fairness_ci_results['tpr_parity_ci'][0],
                            "tpr_parity_ci_upper": fairness_ci_results['tpr_parity_ci'][1],
                            "fnr_parity_ci_lower": fairness_ci_results['fnr_parity_ci'][0],
                            "fnr_parity_ci_upper": fairness_ci_results['fnr_parity_ci'][1],
                            "dp_ratio_ci_lower": fairness_ci_results['dp_ratio_ci'][0],
                            "dp_ratio_ci_upper": fairness_ci_results['dp_ratio_ci'][1],
                            "bestmodel_train_params_dict": bestmodel_train_params,
                            "bestmodel_unique_filename": bestmodel_unique_filename
                        }
                        ci_fairness_mcol.insert_one({**base_res, **fairness_ci_res})

    mclient.close()


if __name__ == "__main__":
    main()
