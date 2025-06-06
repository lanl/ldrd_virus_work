import pandas as pd
import numpy as np


def get_surface_exposure_status(
    virus_names: list[str], protein_names: list[str], surface_exposed_df: pd.DataFrame
) -> list[str]:
    """
    get the surface exposure status of virus-protein pairs from the dataframe of known surface exposed proteins

    Parameters:
    -----------
    virus_names: list
        list of virus names associated with a given kmer feature
    protein_names: list
        list of protein names associated with a given kmer feature
    surface_exposed_df: pd.DataFrame
        dataframe containing the known surface exposure status of virus-protein pairs from the dataset

    Returns:
    --------
    surface_exposed_status: list
        "yes" or "no" status of virus-protein pairs associated with each kmer feature
    """
    # make dataframe of virus-protein pairs from kmer info
    virus_protein_pairs = pd.DataFrame(
        {"virus_names": virus_names, "protein_names": protein_names}
    )
    # get the surface exposure status of kmer virus-protein pairs by merging the two dataframes
    virus_protein_df = pd.merge(
        virus_protein_pairs,
        surface_exposed_df,
        on=["virus_names", "protein_names"],
        how="left",
    )
    surface_exposed_status = list(virus_protein_df["surface_exposed_status"])

    return surface_exposed_status


def merge_tables(train_file: str, igsf_file: str) -> pd.DataFrame:
    """
    merges `igsf_training.csv` and  `receptor_training.csv` dataframes
    and reconciles overlapping entries by aligning accession
    and reassigning values to a new dataframe.

    Parameters:
    -----------
    train_file: str
        file path for loading to `receptor_training.csv`
    igsf_file: str
        file path for loading to `igsf_training.csv`

    Returns:
    --------
    reconciled_df: pd.DataFrame
        merged dataframe with data from train_file and igsf_file and
        reconciled overlapping entries
    """
    # load igsf and receptor datasets
    igsf_training = pd.read_csv(igsf_file)
    receptor_training = pd.read_csv(train_file)

    receptor_training["Receptor_Type"] = receptor_training["Receptor_Type"].str.replace(
        "both", "integrin_sialic_acid"
    )
    reconciled_df = pd.concat([receptor_training, igsf_training], ignore_index=True)
    existing_entries = reconciled_df.duplicated(subset="Accessions", keep="last")
    reconciled_df.loc[existing_entries, "Receptor_Type"] = (
        reconciled_df.loc[existing_entries, "Receptor_Type"] + "_IgSF"
    )
    new_entries = reconciled_df[
        reconciled_df.duplicated(subset="Accessions", keep="first")
    ].index
    reconciled_df = reconciled_df.drop(new_entries).reset_index(drop=True)

    return reconciled_df


def convert_merged_tbl(input_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    convert the merged data table containing both the `receptor_training.csv`
    and `igsf_training.csv` into a format usable by the workflow.

    rename training columns based on receptor type and make new columns
    for overlapping receptor binding targets by combining receptor names

    Parameters
    ----------
    input_tbl : pd.DataFrame
        dataframe containing merged datasets from `receptor_training.csv` and `igsf_training.csv`

    Returns
    -------
    table : pd.DataFrame
        converted dataframe with new columns representing individual and combined receptor targets
    """
    table = input_tbl.drop(
        columns=["Citation_for_receptor", "Whole_Genome", "Mammal_Host", "Primate_Host"]
    )

    table["SA_IG"] = np.where(
        table["Receptor_Type"].isin(["sialic_acid_IgSF", "integrin_sialic_acid_IgSF"]),
        True,
        False,
    )
    table["IN_IG"] = np.where(
        table["Receptor_Type"].isin(["integrin_IgSF", "integrin_sialic_acid_IgSF"]),
        True,
        False,
    )
    table["IN_SA"] = np.where(
        table["Receptor_Type"].isin(
            ["integrin_sialic_acid", "integrin_sialic_acid_IgSF"]
        ),
        True,
        False,
    )
    table["IN_SA_IG"] = np.where(
        table["Receptor_Type"] == "integrin_sialic_acid_IgSF", True, False
    )

    table["SA"] = np.where(
        np.isin(
            table["Receptor_Type"],
            [
                "sialic_acid",
                "integrin_sialic_acid",
                "sialic_acid_IgSF",
                "integrin_sialic_acid_IgSF",
            ],
        ),
        True,
        False,
    )
    table["IG"] = np.where(
        np.isin(
            table["Receptor_Type"],
            ["IgSF", "integrin_IgSF", "sialic_acid_IgSF", "integrin_sialic_acid_IgSF"],
        ),
        True,
        False,
    )
    table["IN"] = np.where(
        np.isin(
            table["Receptor_Type"],
            [
                "integrin",
                "integrin_sialic_acid",
                "integrin_IgSF",
                "integrin_sialic_acid_IgSF",
            ],
        ),
        True,
        False,
    )
    table.rename(
        columns={
            "Virus_Name": "Species",
            "Human_Host": "Human Host",
        },
        inplace=True,
    )
    table.drop(columns="Receptor_Type", inplace=True)

    return table
