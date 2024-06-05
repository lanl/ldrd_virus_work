import pandas as pd
from viral_seq.analysis import spillover_predict as sp
from viral_seq.cli import cli
from importlib.resources import files
from pathlib import Path

data = files("viral_seq.data")
train_data = str(data.joinpath("Mollentze_Training.csv"))
test_data = str(data.joinpath("Mollentze_Holdout.csv"))
cache_viral = Path("data_external/cache_viral")
cache_viral.mkdir(parents=True, exist_ok=True)
train_fixed_data = str(Path("Mollentze_Training_Fixed.csv"))
test_fixed_data = str(Path("Mollentze_Holdout_Fixed.csv"))

train_accessions: set[str] = set()
email = "arhall@lanl.gov"

for file in [train_data, test_data]:
    cli.pull_data(
        [
            "--email",
            email,
            "--cache",
            cache_viral,
            "--file",
            file,
            "--no-filter",
        ],
        standalone_mode=False,
    )


def get_bad_indexes(df, test=False):
    partial_lst = []
    duplicate_lst = []
    no_good_cds_lst = []
    for index, row in df.iterrows():
        accessions = row["Accessions"].split()
        if test and len(train_accessions.intersection(set(accessions))) > 0:
            duplicate_lst.append(index)
        records = sp.load_from_cache(accessions, cache=cache_viral, filter=False)
        partial_bool = False
        bad_cds_bool = False
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


df_train = pd.read_csv(train_data, index_col=0)
train_accessions = set((" ".join(df_train["Accessions"].values)).split())
res_train = get_bad_indexes(df_train)
df_train_fixed = df_train.drop(res_train["partial"] + res_train["no_good_cds"])
df_train_fixed.to_csv(train_fixed_data)

df_test = pd.read_csv(test_data, index_col=0)
res_test = get_bad_indexes(df_test, True)
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

removed_from_train = set(df_train.iloc[res_train["partial"]]["Species"].values)
removed_from_train.update(df_train.iloc[res_train["no_good_cds"]]["Species"].values)

assert df_train_fixed.shape == (
    df_train.shape[0] - len(removed_from_train),
    df_train.shape[1],
), f"{df_train_fixed.shape} should equal ({df_train.shape[0]} - {len(removed_from_train)}, {df_train.shape[1]})"

removed_from_test = set(df_test.iloc[res_test["partial"]]["Species"].values)
removed_from_test.update(df_test.iloc[res_test["no_good_cds"]]["Species"].values)
removed_from_test.update(df_test.iloc[res_test["duplicate"]]["Species"].values)

assert df_test_fixed.shape == (
    df_test.shape[0] - len(removed_from_test),
    df_test.shape[1],
), f"{df_test_fixed.shape} should equal ({df_test.shape[0]} - {len(removed_from_test)}, {df_test.shape[1]})"

assert not df_train_fixed.duplicated(
    subset="Accessions"
).any(), "df_train_fixed contains duplicate accessions"
assert not df_test_fixed.duplicated(
    subset="Accessions"
).any(), "df_test_fixed contains duplicate accessions"

print(
    "Generated 'fixed' datasets with no partials, no genomes without 'good' CDS, and duplicates dropped from test data:",
    train_fixed_data,
    test_fixed_data,
)
