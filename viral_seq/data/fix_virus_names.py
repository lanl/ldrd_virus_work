import pandas as pd
from viral_seq.analysis.spillover_predict import load_results
from pathlib import Path


def fix_virus_names(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # load all accessions
    accessions_dict = {
        s.split(" ")[0]: i for i, s in enumerate(df["Accessions"].values)
    }
    records_unordered = load_results(
        set(accessions_dict.keys()), email="awitmer@lanl.gov"
    )
    out_df = df.copy()
    for record in records_unordered:
        acc = record.id
        org = record.annotations["organism"]
        out_df.at[accessions_dict[acc], column_name] = org
    return out_df


if __name__ == "__main__":
    # import receptor training csv
    receptor_file = Path("receptor_training.csv")
    igsf_file = Path("igsf_training.csv")
    # for each file rename all the viruses to the current virus name stored in the genbank accession
    input_files = [receptor_file, igsf_file]
    for input_file in input_files:
        df = pd.read_csv(input_file)
        out_df = fix_virus_names(df, "Virus_Name")
        out_df.to_csv(input_file, index=False)
