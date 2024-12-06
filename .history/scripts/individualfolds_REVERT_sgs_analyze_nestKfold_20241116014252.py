import argparse
import datetime
import pandas as pd
from pymongo import MongoClient
import urllib.parse


"""
nobal_CHIL_nosmote_sgs_evaluate_baselines_nestKfold.py --> nosmote_CHIL_1_26_results (collection in Mongo) --> TODO: generate csv ___ drop NaNs, include * for vals with <10 folds.
    - precision: NaN for many
    - f1: NaN for many.
    - MODELS: this collection has ALL logr_saga, xgboost, and rf.

bal_CHIL_nosmote_sgs_evaluate_baselines_nestKfold.py --> bal_logr_nosmote_CHIL_2_2_results (coll) --> dropNaNs_bal_CHIL_analyze_nestKfold_nosmote-2-2.csv (results)
    - precision: (demo, white) precision was NaN. All others were defined. 
    - f1: defined for all.
    - drop NaNs, include * for precision vals with <10 folds.
    - MODELS: ONLY has logr_saga.

    - TODO: many Fairness metrics have NaN in "bal_logr_nosmote_CHIL_2_2_fair" (collection)
        - might be okay, remember to drop record if sg_key == "all_patients", because NaN is automatically stored.

        
[current] bal_allfeats_CHIL_nosmote_sgs_evaluate_baselines_nestKfold.py --> bal_allfeats_logr_nosmote_CHIL_2_4_results (coll) --> dropNaNs_bal_allfeats_CHIL_analyze_nestKfold_nosmote-2-4.csv
        
1. Running balanced logr_saga
- feat: (all table feats)
- sgs: only "all_patients", "race"
- county vs tract AHRQ




"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", help="output file",
                        default="final_results_individualfolds_Circ_2024.csv") # Circ_HF_results_individualfolds, ML4H_rebuttal_HF_results_individualfolds, 
    # subgroups: results_clean_tract_ahrq_AQIimp1_baseline_SGs_2.csv
    # Sep subgroups: results_clean_tract_ahrq_AQIimp1_baseline_sep_SGs.csv
    #username = urllib.parse.quote_plus('fensorechase')
    #password = urllib.parse.quote_plus('7pzNiMi7dD!d@Ab')
    parser.add_argument("-mongo_url", default = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.3') # 'mongodb+srv://%s:%s@cluster0.ysobo3u.mongodb.net/' % (username, password)
    parser.add_argument("-mongo_db",
                        default="3aequitas") # 2aequitas
    # 10_4_ACC_agnostic
    parser.add_argument("-mongo_col",
                        default="Circ_HF_results_final", # Circ_HF_results, ml4h_LLMs, Circ_HF_results_DF2
                        help="collection_type") # For subgroup results, set default="subgroups_baseline", for entire dataset, default="baseline"
    args = parser.parse_args()

    # setup the mongo stuff
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]

    pipe_list = [
    {
        "$project": {
            "_id": 0,
            "model": "$model",
            "feat": "$feat",
            "file": "$file",
            "fold": "$fold",
            "endpoint": "$endpoint",
            "sg_key": "$sg_key",
            "subgroup": "$subgroup",
            "auc": "$auc",
            "auc_sd": { "$stdDevSamp": "$auc" },
            "aps": "$aps",
            "aps_sd": { "$stdDevSamp": "$aps" },
            "precision": "$precision",
            "precision_sd": { "$stdDevSamp": "$precision" },
            "recall": "$recall",
            "recall_sd": { "$stdDevSamp": "$recall" },
            "f1": "$f1",
            "f1_sd": { "$stdDevSamp": "$f1" },
            "auprc": "$auprc",
            "auprc_sd": { "$stdDevSamp": "$auprc" },
            "fnr": "$fnr",
            "fnr_sd": { "$stdDevSamp": "$fnr" },
            "tnr": "$tnr",
            "tnr_sd": { "$stdDevSamp": "$tnr" },
            "fpr": "$fpr",
            "fpr_sd": { "$stdDevSamp": "$fpr" },
            "mcc": "$mcc",
            "mcc_sd": { "$stdDevSamp": "$mcc" },
            "n_runs": 1,  # This counts the number of documents
            "test_samp_size": "$test_samp_size"
        }
    }
    ]



    tmp = list(mcol.aggregate(pipe_list))
    tmp_df = pd.DataFrame.from_records(tmp)
    print(tmp_df)
    mclient.close()

    tmp_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

