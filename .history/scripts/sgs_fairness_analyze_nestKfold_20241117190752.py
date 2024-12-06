import argparse
import datetime
import pandas as pd
from pymongo import MongoClient
import urllib.parse


"""

nobal_CHIL_nosmote_sgs_evaluate_baselines_nestKfold.py --> nosmote_CHIL_1_26_fair (coll) --> CHIL_fairness_analyze_nestKfold_nosmote-1-26.csv

bal_CHIL_nosmote_sgs_evaluate_baselines_nestKfold.py --> [TODO: print] bal_logr_nosmote_CHIL_2_2_fair (coll) --> fair_dropNaNs_bal_CHIL_analyze_nestKfold_nosmote-2-2.csv (results)


bal_allfeats_CHIL_nosmote_sgs_evaluate_baselines_nestKfold.py --> bal_allfeats_logr_nosmote_CHIL_2_4_fair (coll) --> fair_dropNaNs_bal_allfeats_CHIL_analyze_nestKfold_nosmote-2-4.csv

"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", help="output file",
                        default="individualfolds_Circ_HF_results_fairness_final.csv") # v3_fairness_circulation_results
    # subgroups: results_clean_tract_ahrq_AQIimp1_baseline_SGs_2.csv
    # Sep subgroups: results_clean_tract_ahrq_AQIimp1_baseline_sep_SGs.csv
    username = urllib.parse.quote_plus('fensorechase')
    password = urllib.parse.quote_plus('7pzNiMi7dD!d@Ab')
    parser.add_argument("-mongo_url", default = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.3') 
    parser.add_argument("-mongo_db",
                        default="3aequitas") # 2aequitas
    # 10_4_ACC_agnostic
    parser.add_argument("-mongo_col",
                        default="Circ_HF_results_fairness_final", # bal_3_5_results_fairness, Circ_HF_results_fairness, ml4h_LLMs
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

            
            "eo_ratio": "$eo_ratio",
            "dp_ratio": "$dp_ratio",
            "fpr_parity": "$fpr_parity",
            "tpr_parity": "$tpr_parity",
            "fnr_parity": "$fnr_parity",


            "total_test_samp_size": "$total_test_samp_size",
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



"""


pipe_list_summarize_folds = [{
    "$group":
            {
                "_id":
                {
                    "feat": "$feat",
                    "model": "$model",
                    "file": "$file",
                    "endpoint": "$endpoint",
                    "sg_key": "$sg_key"
                },

                "eo_ratio":
                {
                    "$avg": "$eo_ratio"
                },
                "eo_ratio_sd":
                {
                    "$stdDevSamp": "$eo_ratio"
                },


                "fpr_parity":
                {
                    "$avg": "$fpr_parity"
                },
                "fpr_parity_sd":
                {
                    "$stdDevSamp": "$fpr_parity"
                },


                "tpr_parity":
                {
                    "$avg": "$tpr_parity"
                },
                "tpr_parity_sd":
                {
                    "$stdDevSamp": "$tpr_parity"
                },


                "fnr_parity":
                {
                    "$avg": "$fnr_parity"
                },
                "fnr_parity_sd":
                {
                    "$stdDevSamp": "$fnr_parity"
                },
            
                

                "total_test_samp_size": 
                {
                    "$avg": "$total_test_samp_size"
                },


               "n_runs":
               {
                   "$sum": 1
                   
               }

            }
    },

    {"$project":
            {
                "_id": 0,
                "model": "$_id.model",
                "feat": "$_id.feat",
                "file": "$_id.file",
                "endpoint": "$_id.endpoint",
                
                "eo_ratio": "$eo_ratio",
                "eo_ratio_sd": "$eo_ratio_sd",

                "fpr_parity": "$fpr_parity",
                "fpr_parity_sd": "$fpr_parity_sd",

                "tpr_parity": "$tpr_parity",
                "tpr_parity_sd": "$tpr_parity_sd",

                "fnr_parity": "$fnr_parity",
                "fnr_parity_sd": "$fnr_parity_sd",


                "sg_key": "$_id.sg_key",
                "total_test_samp_size": "$total_test_samp_size",

                "n_runs": "$n_runs",
                "test_samp_size": "$test_samp_size"
            }
            }
    
    ]


"""