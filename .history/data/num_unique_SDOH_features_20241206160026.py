import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import json

"""
Analyzes and compares unique expanded SDOH features (from AHRQ SDOHD) between county and tract levels from JSON files, removing specific suffixes of features and printing the summary statistics.
"""

def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def remove_level_suffix(feature_list, suffix):
    """
    Remove specified suffix from each feature in the list
    """
    return [feature.replace(suffix, '').strip() for feature in feature_list]

def analyze_feature_uniqueness(base_feat, feat_column, cty_key, trct_key):
    """
    Analyze unique features between county and tract level feature sets
    """
    # First, get the base feature keys from feat_column
    cty_base_keys = feat_column[cty_key]
    trct_base_keys = feat_column[trct_key]

    # Collect all features for county and tract levels
    cty_features = []
    for key in cty_base_keys:
        cty_features.extend(base_feat[key])

    trct_features = []
    for key in trct_base_keys:
        trct_features.extend(base_feat[key])

    # Remove level-specific suffixes
    cty_features_clean = remove_level_suffix(cty_features, "_countylevel")
    trct_features_clean = remove_level_suffix(trct_features, "_census_tractlevel")

    # Convert to sets for comparison
    cty_feature_set = set(cty_features_clean)
    trct_feature_set = set(trct_features_clean)

    # Calculate unique features
    unique_to_county = cty_feature_set - trct_feature_set
    unique_to_tract = trct_feature_set - cty_feature_set
    common_features = cty_feature_set.intersection(trct_feature_set)
    feats_once_or_more = cty_feature_set.union(trct_feature_set)

    print(f"Analysis of features between {cty_key} and {trct_key}:")
    print(f"Base feature keys for county level: {cty_base_keys}")
    print(f"Base feature keys for tract level: {trct_base_keys}")
    print(f"Total county features: {len(cty_features_clean)}")
    print(f"Total tract features: {len(trct_features_clean)}")
    print(f"Features unique to county level: {len(unique_to_county)}")
    print(f"Features unique to tract level: {len(unique_to_tract)}")
    print(f"Common features: {len(common_features)}")
    print(f"Features appearing once or more: {len(feats_once_or_more)}")
    print("\nUnique to County Level:")
    print(sorted(unique_to_county))
    print("\nUnique to Tract Level:")
    print(sorted(unique_to_tract))
    print("\nCommon Features:")
    print(sorted(common_features))
    print("\nFeatures Appearing Once or More:")
    print(sorted(feats_once_or_more))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-base_feat", 
                        default="../data/feat_base.json",
                        help="base features JSON")
    parser.add_argument("-feat_column", 
                        default="../data/feat_column.json",
                        help="feature column JSON")
    
    args = parser.parse_args()

    # Read base features and feature columns
    base_feat = read_json(args.base_feat)
    feat_column = read_json(args.feat_column)

    # Analyze feature uniqueness for specified feature sets
    analyze_feature_uniqueness(base_feat, 
                                feat_column,
                                "M6_total_ahrq_cty_DF1_nm", 
                                "M6_total_ahrq_trct_DF1_nm")

if __name__ == '__main__':
    main()