import argparse
import datetime
import urllib.parse

import pandas as pd
from pymongo import MongoClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-output",
        help="output file",
        default="<your_output_filename>.csv",
    )  # Circ_HF_results_individualfolds, ML4H_rebuttal_HF_results_individualfolds,

    # For local Mongo:
    # Your connection string may look slightly difference. You should copy it from your local MongoDB Compass cluster, after the cluster is started using "mongosh" in the terminal.
    parser.add_argument(
        "-mongo_url",
        default="mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.3",
    )
    parser.add_argument("-mongo_db", default="CIRC_HF_CLUSTER")
    parser.add_argument(
        "-mongo_col", default="<insert_your_collection_name>", help="collection_type" # Your collection name.
    )  # For subgroup results, set default="subgroups_baseline", for entire dataset, default="baseline"
    args = parser.parse_args()

    # setup your mongo info from above
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
                "auc_sd": {"$stdDevSamp": "$auc"},
                "aps": "$aps",
                "aps_sd": {"$stdDevSamp": "$aps"},
                "precision": "$precision",
                "precision_sd": {"$stdDevSamp": "$precision"},
                "recall": "$recall",
                "recall_sd": {"$stdDevSamp": "$recall"},
                "f1": "$f1",
                "f1_sd": {"$stdDevSamp": "$f1"},
                "auprc": "$auprc",
                "auprc_sd": {"$stdDevSamp": "$auprc"},
                "fnr": "$fnr",
                "fnr_sd": {"$stdDevSamp": "$fnr"},
                "tnr": "$tnr",
                "tnr_sd": {"$stdDevSamp": "$tnr"},
                "fpr": "$fpr",
                "fpr_sd": {"$stdDevSamp": "$fpr"},
                "mcc": "$mcc",
                "mcc_sd": {"$stdDevSamp": "$mcc"},
                "n_runs": 1,  # This counts the number of documents
                "test_samp_size": "$test_samp_size",
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
