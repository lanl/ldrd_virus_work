import pandas as pd
import numpy as np
import functools
from importlib.resources import files
import polars as pl
from collections import defaultdict


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
    # TODO: allow for merging multiple data files when adding new viral entries
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


@functools.lru_cache
def _merge_and_convert_tbl(train_file: str, merge_file: str, temp_file: str):
    merge_path = str(files("viral_seq.data").joinpath(merge_file))
    merged_tbl = merge_tables(train_file, merge_path)
    converted_tbl = convert_merged_tbl(merged_tbl)
    converted_tbl.to_csv(temp_file)
    return converted_tbl


def transform_kmer_data(kmer_list: list) -> pd.DataFrame:
    """
    convert list of KmerData objects to dataframe by
    casting each object in the list to a dictionary

    Parameters:
    -----------
    kmer_list: list
        list of KmerData objects

    Returns:
    --------
    kmer_dict_df: pd.DataFrame
        converted and transformed dataframe
    """

    kmer_dict_df = pd.DataFrame([k.__dict__ for k in kmer_list])
    return kmer_dict_df


def get_kmer_viruses(topN_kmers: list, all_kmer_info: pd.DataFrame) -> dict:
    """
    Lookup and store the virus-protein pairs associated with a list of kmers

    Parameters:
    -----------
    topN_kmers: list
        list of kmer features for which to find associated virus-protein pairs
    all_kmer_info: pd.DataFrame
        dataframe holding virus-protein information for kmers

    Returns:
    --------
    kmer_viruses: dict
        dictionary of kmer names and corresponding virus-protein pairs from all_kmer_info
    """
    kmer_viruses = defaultdict(list)
    for kmer in topN_kmers:
        all_kmer_data = all_kmer_info[
            all_kmer_info["kmer_names"].apply(lambda x: x[0]) == kmer
        ]
        for i, kmer_data in all_kmer_data.iterrows():
            if kmer_data is not None:
                kmer_viruses[kmer].append(
                    (kmer_data.virus_name, kmer_data.protein_name)
                )

    return kmer_viruses


def load_kmer_info(file_name: str) -> pd.DataFrame:
    """
    load parquet file containing 'all_kmer_info'

    Parameters:
    -----------
    file_name: str
        file name to load

    Return:
    -------
    all_kmer_info_df: pd.DataFrame
        dataframe containing KmerData class objects
    """
    print(f"Loading {file_name}...")
    all_kmer_info = pl.read_parquet(file_name).to_pandas()
    return all_kmer_info


def save_kmer_info(kmer_info_df: pd.DataFrame, save_file: str) -> None:
    """
    save dataframe containing 'all_kmer_info' to parquet file

    Parameters:
    -----------
    kmer_info_df: pd.DataFrame
        list of KmerData class objects
    """
    kmer_info_df.to_parquet(save_file)
