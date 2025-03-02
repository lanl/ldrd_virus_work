import numpy as np
import pandas as pd
from pathlib import Path
from viral_seq.analysis import spillover_predict as sp
from importlib.resources import files
import tarfile


def make_relabeled_dataset(
    relabeled_data: pd.DataFrame,
    mollentz_train: pd.DataFrame,
    mollentz_test: pd.DataFrame,
) -> tuple:
    targets = relabeled_data.files
    new_dataframe_train = pd.DataFrame()
    new_dataframe_test = pd.DataFrame()
    for t in targets:
        if "train" in t:
            new_dataframe_train[t] = relabeled_data[t]
        elif "test" in t:
            new_dataframe_test[t] = relabeled_data[t]

    new_dataframe_train = new_dataframe_train.rename(
        columns={
            "y_human_train": "human",
            "y_primate_train": "primate",
            "y_mammal_train": "mammal",
        }
    )
    new_dataframe_test = new_dataframe_test.rename(
        columns={
            "y_human_test": "human",
            "y_primate_test": "primate",
            "y_mammal_test": "mammal",
        }
    )

    train_out = pd.concat([mollentz_train, new_dataframe_train], axis=1).drop(
        columns="Human Host"
    )
    test_out = pd.concat([mollentz_test, new_dataframe_test], axis=1).drop(
        columns="Human Host"
    )

    return train_out, test_out


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
    # file paths
    data = files("viral_seq.data")
    train_data = str(data.joinpath("Mollentze_Training.csv"))
    test_data = str(data.joinpath("Mollentze_Holdout.csv"))
    cache_file = str(data.joinpath("cache_mollentze.tar.gz"))
    cache_extract_path = Path("data_external")
    cache_extract_path.mkdir(parents=True, exist_ok=True)
    cache_viral = Path("data_external/cache_viral")

    with tarfile.open(cache_file, "r") as tar:
        tar.extractall(cache_extract_path)

    df_train = pd.read_csv(train_data, index_col=0)
    train_accessions = set((" ".join(df_train["Accessions"].values)).split())
    res_train = get_bad_indexes(df_train, cache_viral)

    df_test = pd.read_csv(test_data, index_col=0)
    res_test = get_bad_indexes(df_test, cache_viral, train_accessions)

    # data structures
    relabeled_data = np.load("relabeled_data.npz")
    mollentz_train = pd.read_csv(train_data).drop(columns="Unnamed: 0")
    mollentz_test = pd.read_csv(test_data).drop(columns="Unnamed: 0")

    # relabel data
    relabeled_df_train, relabeled_df_test = make_relabeled_dataset(
        relabeled_data, mollentz_train, mollentz_test
    )
    relabeled_df_train.drop(
        res_train["partial"] + res_train["no_good_cds"], inplace=True
    )
    relabeled_df_test.drop(
        res_test["partial"] + res_test["no_good_cds"] + res_test["duplicate"],
        inplace=True,
    )

    # save new dataframes
    relabeled_df_train.to_csv("Relabeled_Train.csv", index=False)
    relabeled_df_test.to_csv("Relabeled_Test.csv", index=False)
