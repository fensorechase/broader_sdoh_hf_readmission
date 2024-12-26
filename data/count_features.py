import json


def count_features(json_file, key):
    with open(json_file, "r") as file:
        data = json.load(file)
        if key in data:
            return len(data[key])
        else:
            return 0


def main():
    feat_base_file = "/feat_base.json"
    feat_column_file = "/feat_column.json"

    keys_to_check = [
        "M1",
        "M2",
        "M3_county_DF1_nm",
        "M4_DF1_nm_demo",
        "M5",
        "M5_and_demo",
        "M6_total_ahrq_cty_DF1_nm",
        "M6_total_ahrq_trct_DF1_nm",
        "demo_DF1_nm_county_AHRQ_domain1",
        "demo_DF1_nm_county_AHRQ_domain2",
        "demo_DF1_nm_county_AHRQ_domain3",
        "demo_DF1_nm_county_AHRQ_domain4",
        "demo_DF1_nm_county_AHRQ_domain5",
        "demo_DF1_nm_tract_AHRQ_domain1",
        "demo_DF1_nm_tract_AHRQ_domain2",
        "demo_DF1_nm_tract_AHRQ_domain3",
        "demo_DF1_nm_tract_AHRQ_domain4",
        "demo_DF1_nm_tract_AHRQ_domain5"
    ]  # Replace with actual keys you want to check

    with open(feat_base_file, "r") as base_file:
        base_data = json.load(base_file)

    with open(feat_column_file, "r") as column_file:
        column_data = json.load(column_file)

    for key in keys_to_check:
        base_count = count_features(feat_base_file, key)
        column_count = 0

        if key in column_data:
            for sub_key in column_data[key]:
                if sub_key in base_data:
                    column_count += len(base_data[sub_key])

        print(f"Key: {key}")
        print(f"  Number of features in feat_base.json: {base_count}")
        print(f"  Number of features in feat_column.json: {column_count}")


if __name__ == "__main__":
    main()
