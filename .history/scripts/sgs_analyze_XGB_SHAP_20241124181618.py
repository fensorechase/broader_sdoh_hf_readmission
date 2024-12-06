import argparse
import datetime
import pandas as pd
from pymongo import MongoClient
import urllib.parse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", help="output file",
                        default="Circ_HF_results_xgb_SHAP_final.csv") # Recent: reb_MLHC_final_results_individualfolds, CURRENT other feats: MLHC_results_individualfolds.csv, nanda_MLHC_results_individualfolds.csv
    # Circ_HF_results_xgb_SHAP_final, Circ_HF_results_xgb_SHAP
    # subgroups: results_clean_tract_ahrq_AQIimp1_baseline_SGs_2.csv
    # Sep subgroups: results_clean_tract_ahrq_AQIimp1_baseline_sep_SGs.csv
    username = urllib.parse.quote_plus('fensorechase')
    password = urllib.parse.quote_plus('7pzNiMi7dD!d@Ab')
    parser.add_argument("-mongo_url", default = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.3') 
    # mongodb+srv://%s:%s@cluster0.ysobo3u.mongodb.net/
    # 'mongodb+srv://%s:%s@cluster0.ysobo3u.mongodb.net/' % (username, password)) 
    parser.add_argument("-mongo_db",
                        default="3aequitas") # 2aequitas
    # 10_4_ACC_agnostic
    parser.add_argument("-mongo_col",
                        default="Circ_HF_results_xgb_SHAP_final", # MLHC_final_results MLHC_resolved_results, Circ_HF_results_xgb_SHAP_final, Circ_HF_results_xgb_SHAP
                        help="collection_type") # For subgroup results, set default="subgroups_baseline", for entire dataset, default="baseline"
    args = parser.parse_args()

    # setup the mongo stuff
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]

    records = mcol.find({}, {
        'shap_ordered_names': 1,
        'shap_ordered_importance': 1,
        'model': 1,
        'feat': 1,
        'sg_key': 1,
        'subgroup': 1,
        'fold': "$fold",
        'file': 1,
        'endpoint': 1,
        'auc': 1,
        'aps': 1,
        'precision': 1,
        'recall': 1,
        'f1': 1,
        'auprc': 1,
        'fnr': 1,
        'tnr': 1,
        'fpr': 1,
        'test_samp_size': 1
    })

    data = []
    for record in records:
        shap_names = record.get('shap_ordered_names', [])
        shap_importances = record.get('shap_ordered_importance', [])
        for name, importance in zip(shap_names, shap_importances):
            data.append({
                'shap_ordered_name': name,
                'shap_ordered_importance': importance,
                'model': record.get('model'),
                'feat': record.get('feat'),
                'sg_key': record.get('sg_key'),
                'subgroup': record.get('subgroup'),
                'fold': record.get('fold'),
                'file': record.get('file'),
                'endpoint': record.get('endpoint'),
                'auc': record.get('auc'),
                'aps': record.get('aps'),
                'precision': record.get('precision'),
                'recall': record.get('recall'),
                'f1': record.get('f1'),
                'auprc': record.get('auprc'),
                'fnr': record.get('fnr'),
                'tnr': record.get('tnr'),
                'fpr': record.get('fpr'),
                'test_samp_size': record.get('test_samp_size')
            })

    # create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    mclient.close()


if __name__ == "__main__":
    main()

