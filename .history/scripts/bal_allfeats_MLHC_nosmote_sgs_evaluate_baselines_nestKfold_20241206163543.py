"""
This file is a driver to run binary classification models for specified heart failure (HF) endpoint. Prints results (performance, fairness, feature importance/coefficients) to specificed local MongoDB collection.

"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from datetime import datetime
import json
import pandas as pd
from pymongo import MongoClient
import sklearn.ensemble as sken
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as skms
import sklearn.neighbors as skknn
import sklearn.neural_network as sknn
import sklearn.metrics as skm
import sklearn.tree as sktree
import tqdm
import urllib.parse
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump, load


# For xgboost balanced weights
from sklearn.utils.class_weight import compute_sample_weight
# from focal_loss import BinaryFocalLoss
from imblearn.over_sampling import SMOTE



# from sklearn.inspection import permutation_importance # For feature importance
import shap
import imblearn
from evalHelper import read_json, evaluate_results, get_train_test, evaluate_results_fairness

MODEL_PARAMS = {

    "xgb": 
    {
        "model": xgb.XGBClassifier(),
        "params": {"max_depth": [6],
                   "n_estimators": [50, 500],
                   "learning_rate":[0.01],
                   "eval_metric":["logloss"],
                   "lambda": [1],
                   "alpha": [0, 0.2],
                }
    },

     "rf":
    {
        'model': sken.RandomForestClassifier(),
         'params': {'max_depth': [6, 8],
                    'min_samples_leaf': [5, 10],
                    'n_estimators': [5, 10],
                   'class_weight': ["balanced"]
                }
    },

     "logr":  # lbfgs solver is used.
    {
        "model": LogisticRegression(),
        "params": {"penalty": ['l2'], 
                   "max_iter": [2000], 
                   "solver": ['lbfgs'],
                   "class_weight": ["balanced"]
                }
    }


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
    


def main():
    parser = argparse.ArgumentParser()

    # For local Mongo:
    # Your connection string may look slightly difference. You should copy it from your local MongoDB Compass cluster, after the cluster is started using "mongosh" in the terminal.
    parser.add_argument("-mongo_url", default = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.3')

    parser.add_argument("-mongo_db",
                        default="3aequitas")
    parser.add_argument("-mongo_col",
                        default="Circ_HF_results_final", # Circ_HF_results_DF1
                        help="collection_type") # Used to be subgroups_baseline
    # default information
    # Smaller sample, removed NAs: clean_tract_v2_ahrq.csv: includes select built env, mobility, air quality AHRQ data
    # AHRQ baseline year, removed NAs: clean_tract_v2_ahrq_baseline.csv
    # AHRQ all 30k samples, mean from 2009-2020: clean_tract_v2_ahrq_allsamps.csv
    # AHRQ all 30k samples, baseline year, imputation: clean_tract_v2_ahrq_all_samps_baseline.csv
    # Old input: final_clean_tract_ahrq_AQIimp1_baseline.csv, 10-4-total_feats.cs
    parser.add_argument("-data_file", 
                        default="../data/Circ_final_allstates_modelinput.csv", # Circ_allstates_includemissing_modelinput_DF1.csv, Circ_allstates_EXCLUDEmissing_modelinput_DF2.csv
                        help="data file") 
    parser.add_argument("-base_feat",
                        default="../data/feat_base.json",
                        help="base_features")
                        
    parser.add_argument("-feat_file", 
                        default="../data/feat_column.json",
                       help="model_features")
    
    parser.add_argument("-subgroup_file", 
                    default="../data/subgroup_cols_fast.json",
                    help="subgroups_to_test") 
    parser.add_argument("-endpoint",
                        default="readmit30bin") # readmit30bin
    

    # See if adding AHRQ + EPA_AQS whether 
    #   or not the models would pick up 
    #   the EPA variables before the AHRQ EPAA ones


    # total county medianIMP AHRQ KEEP EPAA medianIMP + EPA_AQS
    # total county medianIMP AHRQ *minus* EPAA + EPA_AQS

    # "county_AHRQ_median_without_EPAA", # total w/o EPAA, already done.

    # total (includes EPAA) + EPA_AQS_median
    # total (includes EPAA) + EPA_AQS_spatial
    # total - EPAA + EPA_AQS_median
    # total - EPAA + EPA_AQS_spatial

    """
        "HFtype_CCIraw_Demo", "AHRQ_HFtype_CCIraw_Demo", 
            "SVI_HFtype_CCIraw_Demo", "ADI_HFtype_CCIraw_Demo", 
            "SDI_HFtype_CCIraw_Demo"

        Next: "demo", "age_current", "sex", "charlson", "comorb", "hf_vars"

    """

    # TODO: For v2.0, we combine "demo" + "clin" into 1 set ("our_baseline_clin_AND_demo")



    """
    **** TODO
    I should use the FULL ahrq cty & tract sets.
        ... not just the "no-missing" sets, like below.
    ******
    """


    """ ************************************
        TODO: include demo...
    "our_baseline_clin_AND_svi",
    "our_baseline_clin_AND_adi_national",
    "our_baseline_clin_AND_sdi",
    """
    parser.add_argument("--feats", nargs='+', default=[
  
        "competition_non_SDOH",
        "M2",
        "M3_county_DF1_nm",
        "M4_DF1_nm_demo", # Demo has 7 vars.
        "M5_and_demo"
        "M6_total_ahrq_cty_DF1_nm", "M6_total_ahrq_trct_DF1_nm",

        "DF1_nm_county_AHRQ_domain1",
        "DF1_nm_county_AHRQ_domain2",
        "DF1_nm_county_AHRQ_domain3",
        "DF1_nm_county_AHRQ_domain4",
        "DF1_nm_county_AHRQ_domain5",

        "DF1_nm_tract_AHRQ_domain1",
        "DF1_nm_tract_AHRQ_domain2",
        "DF1_nm_tract_AHRQ_domain3",
        "DF1_nm_tract_AHRQ_domain4",
        "DF1_nm_tract_AHRQ_domain5"
    
        ])
    

    """

        "our_baseline_clin_AND_demo",

        "ahrq1_tract",
        "ahrq3_tract",
        "ahrq4_tract",

        "resolved_nanda1_tract",
        "resolved_nanda3_tract",
        "resolved_nanda4_tract",


        "1_clin_AHRQ_NaNDA",
        "3_clin_AHRQ_NaNDA",
        "4_clin_AHRQ_NaNDA",

        "AHRQ_TOTAL_tract_and_NANDA_TOTAL_tract_resolved", # ALL SDOH: Note: no AHRQ 2, 5
        "blad_AHRQ_TOTAL_tract_and_NANDA_TOTAL_tract_resolved" # ALL SDOH + ALL CLIN: Note: no AHRQ 2, 5


        # TODO:...
       # "1_ahrq_nanda_resolved", 
       # "3_ahrq_nanda_resolved",
       # "4_ahrq_nanda_resolved",

       # "NANDA_TOTAL_tract_resolved", # NaNDA: all.
       # "AHRQ_TOTAL_tract_no25", # AHRQ: all -- TODO: include 2, 5?


    ###############################
    "our_baseline_clin_AND_demo",

        
        "AHRQ_tract_and_blad",
         "NANDA_tract_and_blad",

        "blad_AHRQ_TOTAL_tract_and_NANDA_TOTAL_tract",


       "AHRQ_TOTAL_tract",
        "NANDA_TOTAL_tract",
        "AHRQ_TOTAL_tract_and_NANDA_TOTAL_tract"


        "ahrq1_tract",
       "ahrq2_tract",
       "ahrq3_tract",
       "ahrq4_tract",
       "ahrq5_tract",

        "nanda1_tract",
       "nanda2_tract",
       "nanda3_tract",
       "nanda4_tract"

    """


    """

        "NaNDA_tract_pollution_d16",

        "NaNDA_tract_schools_d22",
        
        "our_baseline_clin_AND_demo",

        "competition_non_SDOH",
        "competition_SDOH_vars",

        "total_ahrq_cty_nomissing", 
        "total_ahrq_trct_nomissing",

        "M4",
        "M5",
        "M4_and_NaNDA"


    INDIVIDUAL 4 SETS (from v2.0): 


    "competition_non_SDOH",
    "competition_SDOH_vars",

    "ahrq1_social_cty_nomissing",
    "ahrq2_economic_cty_nomissing",
    "ahrq3_education_cty_nomissing",
    "ahrq4_physicalinfrastructure_cty_nomissing",
    "ahrq5_healthcarequality_cty_nomissing",

    "total_ahrq_cty_nomissing", 

    "ahrq1_social_trct_nomissing",
    "ahrq2_economic_trct_nomissing",
    "ahrq3_education_trct_nomissing",
    "ahrq4_physicalinfrastructure_trct_nomissing",
    "ahrq5_healthcarequality_trct_nomissing",

    "total_ahrq_trct_nomissing",

    "our_baseline_clin_AND_demo"
    """


    """
    2-4 Run:
        "sdi_vars", "svi", "svi_RPL_raw_themes"

 "demo", "adi_national", "adi_state", "sdi", "competition_SDOH_vars", "total_ahrq_cty_nomissing", "competition_SDOH_vars_AND_total_ahrq_cty_nomissing",
    
    "our_baseline_clin", "competition_non_SDOH",

    


    "our_baseline_clin_AND_competition_SDOH_vars_AND_total_ahrq_cty_nomissing",

    "sdi_vars", "svi", "svi_RPL_raw_themes",

    "demo", "adi_national", "adi_state", "sdi", "competition_SDOH_vars", "competition_SDOH_vars_AND_total_ahrq_cty_nomissing",
    
    "our_baseline_clin", "competition_non_SDOH",



    "our_baseline_clin_AND_sdi_vars",
    "our_baseline_clin_AND_sdi",
    "our_baseline_clin_AND_adi_national",
    "our_baseline_clin_AND_adi_state",
    "our_baseline_clin_AND_svi",
    "our_baseline_clin_AND_svi_RPL_raw_themes",
    "our_baseline_clin_AND_demo",
    "our_baseline_clin_AND_total_ahrq_cty_nomissing",
    "our_baseline_clin_AND_competition_SDOH_vars",

    "ahrq1_social_cty_nomissing",
    "ahrq2_economic_cty_nomissing",
    "ahrq3_education_cty_nomissing",
    "ahrq4_physicalinfrastructure_cty_nomissing",
    "ahrq5_healthcarequality_cty_nomissing",


    "ahrq1_social_trct_nomissing",
    "ahrq2_economic_trct_nomissing",
    "ahrq3_education_trct_nomissing",
    "ahrq4_physicalinfrastructure_trct_nomissing",
    "ahrq5_healthcarequality_trct_nomissing",

    "total_ahrq_trct_nomissing",
    "total_ahrq_cty_nomissing",



    """









    """
    Re-insert: Feb 2nd removed temporarily:
        "demo", "adi_national", "adi_state", "sdi", "competition_SDOH_vars", "total_ahrq_cty_nomissing", "competition_SDOH_vars_AND_total_ahrq_cty_nomissing",
    
    "our_baseline_clin", "competition_non_SDOH",


    
    
    """
    
    """
    "our_baseline_clin_AND_sdi_vars",
    "our_baseline_clin_AND_sdi",
    "our_baseline_clin_AND_adi_national",
    "our_baseline_clin_AND_adi_state",
    "our_baseline_clin_AND_svi",
    "our_baseline_clin_AND_svi_RPL_raw_themes",
    "our_baseline_clin_AND_demo",
    "our_baseline_clin_AND_total_ahrq_cty_nomissing",
    "our_baseline_clin_AND_competition_SDOH_vars",



    "ahrq1_social_cty_nomissing",
    "ahrq2_economic_cty_nomissing",
    "ahrq3_education_cty_nomissing",
    "ahrq4_physicalinfrastructure_cty_nomissing",
    "ahrq5_healthcarequality_cty_nomissing"




    Most recent run:
     "competition_non_SDOH",
    "our_baseline_clin",

    "competition_SDOH_vars",
    "sdi_vars", "sdi", 
    "adi_national", "adi_state",
    "svi", "svi_RPL_raw_themes",
    "demo", "total_ahrq_cty_nomissing",

    "our_baseline_clin_AND_sdi_vars",
    "our_baseline_clin_AND_sdi",
    "our_baseline_clin_AND_adi_national",
    "our_baseline_clin_AND_adi_state",
    "our_baseline_clin_AND_svi",
    "our_baseline_clin_AND_svi_RPL_raw_themes",
    "our_baseline_clin_AND_demo",
    "our_baseline_clin_AND_total_ahrq_cty_nomissing",
    "our_baseline_clin_AND_competition_SDOH_vars"

    
    """



    """
    ------ TODO: re-run in same Mongo collection. ------

        "ahrq1_social_cty_nomissing",
        "ahrq2_economic_cty_nomissing",
        "ahrq3_education_cty_nomissing",
        "ahrq4_physicalinfrastructure_cty_nomissing",
        "ahrq5_healthcarequality_cty_nomissing",

    ------ end TODO: -------
    
    """
    
    """
        "ahrq1_social_cty_nomissing",
        "ahrq2_economic_cty_nomissing",
        "ahrq3_education_cty_nomissing",
        "ahrq4_physicalinfrastructure_cty_nomissing",
        "ahrq5_healthcarequality_cty_nomissing",

        "total_ahrq_cty_nomissing",

        "ahrq1_social_trct_nomissing",
        "ahrq2_economic_trct_nomissing",
        "ahrq3_education_trct_nomissing",
        "ahrq4_physicalinfrastructure_trct_nomissing",
        "ahrq5_healthcarequality_trct_nomissing",

        "total_ahrq_trct_nomissing",

        "sdi_vars", "sdi", 
        "adi_national", "adi_state",
        "svi", "svi_RPL_raw_themes"
    
    
    """
   
   # --- todo: make json with subgroup column names, then run TEST results for each col names in the json. 
   # ... similar to how (models+input file) print diff column names in test results


    args = parser.parse_args()

    # setup mongo
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]
    fairness_mcol = mdb["Circ_HF_results_fairness_final"] # Used to be: "bal_allfeats_logr_nosmote_CHIL_2_4_fair"
    # raw_mcol = mdb["smote_CHIL_1_20_raw_preds"] # Used to be: "raw_baseline"
    logcoeffs_mcol = mdb["Circ_HF_results_log_coeffs_final"] # To save logisitic coeffs: "bal_allfeats_logr_nosmote_CHIL_2_4_log_coeffs"
    #logCI_mcol = mdb["9_10_MLH_log_CI"] # For 95% CI.
    shap_mcol = mdb["Circ_HF_results_xgb_SHAP_final"] # To save shap for XGBoost: smote_CHIL_1_20_xgb_SHAP

    df = pd.read_csv(args.data_file)
    base_feat = read_json(args.base_feat)
    feat_info = read_json(args.feat_file)
    subgroups_bins = read_json(args.subgroup_file)

    # determine the feature sets
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
    # then, access exact subgroup name: "for g in subgroup_bins[s]"
    subgroup_keys = subgroups_bins.keys()



    for i in tqdm.tqdm(range(1, 11), desc="test-split"):
        train_df, test_df, train_y, test_y = get_train_test(df, i, label=args.endpoint)
        
        test_y = test_y.astype(int)  # Cast to ensure labels are integers

        # Reset test_df indices for subgroup indexing
        test_df = test_df.reset_index()

        
        for fname, fcolumns in tqdm.tqdm(feat_cols.items(),
                                         desc="feats", leave=False):
            base_res = {
                "file": args.data_file,
                "feat": fname,
                "endpoint": args.endpoint,
                "fold": i
            }

            # for both train and test get only those columns
            train_x = train_df[fcolumns]

            # Apply imputation: 
            imputer = SimpleImputer(missing_values = np.nan, strategy='median') #, keep_empty_features=True
            train_x = imputer.fit_transform(train_x)

            # smt = SMOTE(random_state=42)
            # train_x, smt_train_y = smt.fit_resample(train_x, train_y)

            # Apply feature preprocessing: StandardScaler
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            
            # logr = LogisticRegression(class_weight="balanced" , penalty="none", max_iter=10000, solver='saga') # Used to be 2000 for 9-6, often didn't converge.
            # Maybe not converging for logistic because of multicollinearity?
            


            for mname, mk_dict in tqdm.tqdm(MODEL_PARAMS.items(),
                                            desc="models", leave=False):
                
                gs = skms.GridSearchCV(mk_dict["model"],
                                       mk_dict["params"],
                                       cv=5,
                                       n_jobs=4,
                                       scoring='roc_auc', refit=True)
                
                if mname != 'xgb':
                    gs.fit(train_x, train_y)  
                else:
                    gs.fit(train_x, train_y, sample_weight=compute_sample_weight("balanced", train_y))

                # best_val_rocauc = gs.best_score_ # Float.
                # best_val_params = gs.best_params_ # This is a DICT.

                # Save model file for later use: 

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

                ##### TODO - uncomment to save model: dump(model, bestmodel_unique_filename) 


               
                # Loop through test eval for each subgroup
                for s in subgroup_keys:
                    # 1 entire dict is for 1 SET of subgroups
                    # (e.g., 1 dict for "race" subgroups, since "s" is "race" one iteration.)
                    subgroup_preds_dict = {} # curr_subgroup: [test_x, sg_test_y]
                    # Loop through test eval for each subgroup

                    for g in subgroups_bins[s]:
                        # Get indices of rows that match curr_subgroup
                        cs_ind = test_df.loc[test_df[g]==1].index
                        # For subgroups, select only current 'sg' from test_df & test_y.
                        sg_test_df = test_df[test_df[g] == 1]
                        sg_test_y = test_y.iloc[cs_ind] # use column indices for test_y, bc no subgroup cols here
                        
                        print("______________________________________")
                        print("FOLD: ", i, "MODEL: ", mname)

                        #print("FOLD: ", i, "LEN SG_TEST_Y (true vals)", len(sg_test_y), " ", g) # all_patients should be 2899.
                        #print("FOLD: ", i, "0 SG_TEST_Y (true vals)", (sg_test_y == 0).sum(), " ", g) # all_patients should be 2398.
                        #print("FOLD: ", i, "1 SG_TEST_Y (true vals)", (sg_test_y == 1).sum(), " ", g) # all_patients should be 501.


                        # Get only desired feature cols
                        test_x = sg_test_df[fcolumns]
                        # get the test encounter id
                        test_idx = sg_test_df["Encounter"]
                        auc, aps, y_hat, binary_predictions, precision, recall, f1, auc_precision_recall, fnr,  tnr, fpr, mcc = evaluate_results(gs, test_x, sg_test_y, args.endpoint, imputer=imputer, scaler=scaler)
                        
                        # print("LEN BINARY PREDS (output of models)", len(binary_predictions), g)

                        
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
                            "tnr":  tnr, 
                            "fpr":  fpr,
                            'mcc': mcc,
                            "sg_key": s,
                            "subgroup": g,
                            "test_samp_size": len(sg_test_y),
                            "bestmodel_train_params_dict": bestmodel_train_params,
                            "bestmodel_unique_filename": bestmodel_unique_filename

                        }
                        mcol.insert_one({**base_res, **perf_res})
                        

                        # Get SHAP for XGBoost
                        if mname == "xgb":
                            # Answer (ntnq): https://datascience.stackexchange.com/questions/52476/how-to-use-shap-kernal-explainer-with-pipeline-models
                            # model = gs.best_estimator_ # Best XGBoost model.

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
                                "tnr":  tnr, 
                                "fpr":  fpr,
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
                        if "logr" in mname: # mname == "logr_lbfgs" or mname == "logr_saga":
                            # Note: model is a pipeline. Select further beyond this.
                            # model = gs.best_estimator_ #[-1] if pipeline. Best LogisticRegression model.
                            
                            # Save logr raw preds, & log coeffs
                            #tmp = dict(zip(test_idx, y_hat))
                            #raw_res = {
                            #    "model": mname,
                            #    "pred": json.dumps(tmp)
                            #}
                            #raw_mcol.insert_one({**base_res, **raw_res})
                            # Save logistic coeffs for: black, white subgroups specifically.
                            # Save these for EACH set of feats (track which 'feat' and 'subgroup')
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
                                "tnr":  tnr, 
                                "fpr":  fpr,
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
                    # NOTE: this currently only works for 1 protected characterstic (e.g., race)
                    # After subgroup loop, calculate fairness:
                    eo_ratio, fpr_parity, tpr_parity, fnr_parity, dpr = evaluate_results_fairness(gs, subgroup_preds_dict)
                    
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
                        
                        

    mclient.close()


if __name__ == '__main__':
    main()

