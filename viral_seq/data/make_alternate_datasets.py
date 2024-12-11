import pandas as pd
from viral_seq.analysis import spillover_predict as sp
from importlib.resources import files
from pathlib import Path
import tarfile


def get_bad_indexes(df, cache, train_accessions=None):
    partial_lst = []
    duplicate_lst = []
    no_good_cds_lst = []
    for index, row in df.iterrows():
        accessions = row["Accessions"].split()
        if (
            train_accessions is not None
            and len(train_accessions.intersection(set(accessions))) > 0
        ):
            duplicate_lst.append(index)
        records = sp.load_from_cache(accessions, cache=cache, filter=False)
        partial_bool = False
        good_cds_bool = False
        for record in records:
            if "partial" in record.description:
                partial_bool = True
            for feature in record.features:
                if feature.type == "CDS":
                    nuc_seq = feature.location.extract(record.seq)
                    if len(nuc_seq) % 3 == 0:
                        good_cds_bool = True
        if partial_bool:
            partial_lst.append(index)
        if not good_cds_bool:
            no_good_cds_lst.append(index)
    return {
        "partial": partial_lst,
        "duplicate": duplicate_lst,
        "no_good_cds": no_good_cds_lst,
    }


if __name__ == "__main__":
    data = files("viral_seq.data")
    train_data = str(data.joinpath("Mollentze_Training.csv"))
    test_data = str(data.joinpath("Mollentze_Holdout.csv"))
    cache_file = str(data.joinpath("cache_mollentze.tar.gz"))
    cache_extract_path = Path("data_external")
    cache_extract_path.mkdir(parents=True, exist_ok=True)
    cache_viral = Path("data_external/cache_viral")
    train_fixed_data = Path("Mollentze_Training_Fixed.csv")
    test_fixed_data = Path("Mollentze_Holdout_Fixed.csv")
    train_accessions: set[str] = set()

    with tarfile.open(cache_file, "r") as tar:
        tar.extractall(cache_extract_path)

    df_train = pd.read_csv(train_data, index_col=0)
    train_accessions = set((" ".join(df_train["Accessions"].values)).split())
    res_train = get_bad_indexes(df_train, cache_viral)
    df_train_fixed = df_train.drop(res_train["partial"] + res_train["no_good_cds"])
    df_train_fixed.to_csv(train_fixed_data)

    df_test = pd.read_csv(test_data, index_col=0)
    res_test = get_bad_indexes(df_test, cache_viral, train_accessions)
    df_test_fixed = df_test.drop(
        res_test["partial"] + res_test["no_good_cds"] + res_test["duplicate"]
    )
    df_test_fixed.to_csv(test_fixed_data)

    print("=== Train Summary ===")

    print("Partial:", df_train.iloc[res_train["partial"]]["Species"].values)
    print("No Good CDS:", df_train.iloc[res_train["no_good_cds"]]["Species"].values)

    print("=== Test Summary ===")
    print("Partial:", df_test.iloc[res_test["partial"]]["Species"].values)
    print("No Good CDS:", df_test.iloc[res_test["no_good_cds"]]["Species"].values)
    print("In training:", df_test.iloc[res_test["duplicate"]]["Species"].values)

    print(
        "Generated 'fixed' datasets with no partials, no genomes without 'good' CDS, and duplicates dropped from test data:",
        train_fixed_data,
        test_fixed_data,
    )
