import pandas as pd
from viral_seq.analysis.spillover_predict import load_results
from pathlib import Path


def fix_virus_names(df: pd.DataFrame, new_names: pd.DataFrame, column_name: str):
    """
    rename viruses in given dataset based on mappings from previous name to new name

    Parameters:
    -----------
    df: pd.DataFrame
            dataframe for which to rename values
    new_names: pd.DataFrame
            dataframe containing mapping between old names and new names
    column_name: str
            column name in df to replace names

    Returns:
    --------
    new_df: pd.DataFrame
            dataframe containing renamed viruses
    """
    # rename viruses
    new_df = df.copy()
    for i, row in df.iterrows():
        virus_name = row[column_name]
        match_row = new_names.loc[new_names["old_name"] == virus_name]
        if match_row.empty:
            continue
        # in this case, find the first instance of the old name and then replace both instances
        elif match_row.shape[0] > 1:
            single_row = match_row.iloc[0]
            old_name = single_row["old_name"]
            new_name = single_row["new_name"]
        else:
            old_name = match_row["old_name"].item()
            new_name = match_row["new_name"].item()
        new_df["Virus_Name"] = new_df["Virus_Name"].replace(old_name, new_name)
    return new_df


if __name__ == "__main__":
    # import receptor training csv
    receptor_file = Path("receptor_training.csv")
    igsf_file = Path("igsf_training.csv")

    input_files = [receptor_file, igsf_file]
    for input_file in input_files:
        training_df = pd.read_csv(input_file)
        # load all accessions
        records_dict: dict[str, list] = {}
        accessions_dict: dict[str, str] = {}
        row_dict: dict[str, pd.Series] = {}
        accessions = []
        new_names = pd.DataFrame(columns=["old_name", "new_name"])
        for index, row in training_df.iterrows():
            records_dict[row["Virus_Name"]] = []
            row_dict[row["Virus_Name"]] = row
            for accession in row["Accessions"].split():
                accessions.append(accession)
                accession_key = accession.split(".")[0]
                accessions_dict[accession_key] = row["Virus_Name"]
                # load all records on the fly
            records_unordered = load_results(set(accessions), email="awitmer@lanl.gov")
            for record in records_unordered:
                records_dict[accessions_dict[record.id.split(".")[0]]].append(record)

        # find previous name in dataset and match it to new name from viral record
        for record in records_unordered:
            acc = record.id
            org = record.annotations["organism"]
            old_name = training_df.loc[
                training_df["Accessions"].str.contains(record.id)
            ]["Virus_Name"].item()
            if old_name != org:
                new_row = {"old_name": old_name, "new_name": org}
                new_names = pd.concat(
                    [new_names, pd.DataFrame([new_row])], ignore_index=True
                )
        # drop duplicate entries in the dataframe containing matching names
        new_names.drop_duplicates(subset=["old_name", "new_name"], inplace=True)

        # rename 'receptor_training' viruses
        output_df = fix_virus_names(training_df, new_names, "Virus_Name")
        output_df.to_csv(input_file, index=False)
