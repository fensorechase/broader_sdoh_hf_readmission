import argparse
import datetime
import urllib.parse

import pandas as pd
from pymongo import MongoClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-output", help="output file", default="Circ_HF_results_xgb_SHAP_final.csv"
    )
    # For local Mongo:
    # Your connection string may look slightly difference. You should copy it from your local MongoDB Compass cluster, after the cluster is started using "mongosh" in the terminal.
    parser.add_argument(
        "-mongo_url",
        default="mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.3",
    )

    parser.add_argument("-mongo_db", default="CIRC_HF_CLUSTER")
    # 10_4_ACC_agnostic
    parser.add_argument(
        "-mongo_col", default="Circ_HF_results_xgb_SHAP_final", help="collection_type"
    )
    args = parser.parse_args()

    # setup the mongo info
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]

    records = mcol.find(
        {},
        {
            "shap_ordered_names": 1,
            "shap_ordered_importance": 1,
            "model": 1,
            "feat": 1,
            "sg_key": 1,
            "subgroup": 1,
            "fold": "$fold",
            "file": 1,
            "endpoint": 1,
            "auc": 1,
            "aps": 1,
            "precision": 1,
            "recall": 1,
            "f1": 1,
            "auprc": 1,
            "fnr": 1,
            "tnr": 1,
            "fpr": 1,
            "test_samp_size": 1,
        },
    )

    data = []
    for record in records:
        shap_names = record.get("shap_ordered_names", [])
        shap_importances = record.get("shap_ordered_importance", [])
        for name, importance in zip(shap_names, shap_importances):
            data.append(
                {
                    "shap_ordered_name": name,
                    "shap_ordered_importance": importance,
                    "model": record.get("model"),
                    "feat": record.get("feat"),
                    "sg_key": record.get("sg_key"),
                    "subgroup": record.get("subgroup"),
                    "fold": record.get("fold"),
                    "file": record.get("file"),
                    "endpoint": record.get("endpoint"),
                    "auc": record.get("auc"),
                    "aps": record.get("aps"),
                    "precision": record.get("precision"),
                    "recall": record.get("recall"),
                    "f1": record.get("f1"),
                    "auprc": record.get("auprc"),
                    "fnr": record.get("fnr"),
                    "tnr": record.get("tnr"),
                    "fpr": record.get("fpr"),
                    "test_samp_size": record.get("test_samp_size"),
                }
            )

    # create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    mclient.close()


if __name__ == "__main__":
    main()
