import pandas as pd
from typing import Optional
import numpy as np
import functools
from importlib.resources import files
from typing import Union, Dict
import polars as pl


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


def find_matching_kmers(target_column: str, mapping_methods: list) -> str:
    """
    find the matching AA-kmers between workflow runs using different PC mapping methods.
    return the appropriate string response based on the possible outputs of the function,
    which are:
        1. the workflow has not been run with both mapping methods
        2. no matching kmers are found between the two mapping methods
        3. matching kmers are found between the two mapping methods

    Parameters:
    -----------
    target_column: str
        target column string for collecting correct features
    mapping_methods: str
        list of mapping methods between which to check for matching features
    Returns:
    --------
    str
        one of three different output strings corresponding to outcome of the
        search for matching kmers
    """
    # check to see if the necessary files are available and, if so, load them and perform kmer matching
    try:
        print(
            f"Will try to load topN kmer files for {mapping_methods} mapping schemes..."
        )
        topN_files = []
        for mm in mapping_methods:
            topN_file = pl.read_parquet(
                f"topN_kmers_{target_column}_{mm}.parquet.gzip"
            ).to_pandas()
            topN_files.append(topN_file)
        kmer_matches = match_kmers(topN_files, mapping_methods)
        if kmer_matches is not None and not kmer_matches.empty:
            return f"Matching AA kmers between PC kmers:\n {kmer_matches.to_string(index=False)}"
        else:
            return "No matching AA kmers found in TopN."
    except FileNotFoundError:
        return_statement = "Must run workflow using both mapping methods before performing kmer mapping."
    return return_statement


def match_kmers(
    topN_files: list[pd.DataFrame],
    mapping_methods: list[str],
    save_dir: str = "kmer_maps",
) -> Optional[pd.DataFrame]:
    """
    determine matching AA kmers between different PC mappings from topN kmer features
    using each mapping method (i.e. ``jurgen_schmidt`` and ``shen_2007``). If matches
    exists, return dataframe containing matching AA kmer and corresponding PC kmer from
    each mapping method.

    in order for this function to execute, you must run the full workflow using both
    `--mapping-method` arguments to generate the required kmer mapping files

    Parameters:
    -----------
    topN_files: list[pd.DataFrame]
        list of pandas dataframes containing topN kmers found using each mapping scheme
    mapping_methods: list[str]
        names of mapping methods to compare
    save_dir: str
        file name for saving kmer_maps

    Return:
    -------
    kmer_matches_df: Optional[pd.DataFrame]
        dataframe of PC kmer matches and their associated AA-kmer
    """
    topN_str_count = []
    kmer_lists = []
    for topN_file in topN_files:
        topN_list = list(topN_file["0"])
        kmer_lists.append(topN_list)

        # find lengths of topN kmers strings in order to load correct PC-AA kmer map parquet files
        topN_str = [
            s.replace("kmer_PC_", "").replace("kmer_AA_", "") for s in topN_list
        ]
        topN_str_count.extend([len(s) for s in topN_str])

    topN_kmer_lengths = list(set(topN_str_count))

    # load appropriate kmer-matching files
    kmer_maps_list = []
    for mm in mapping_methods:
        for kmer_len in topN_kmer_lengths:
            kmer_df = pl.read_parquet(
                f"{save_dir}/kmer_maps_k{kmer_len}_{mm}.parquet.gzip"
            ).to_pandas()
            kmer_maps_list.append(kmer_df)
    kmer_maps_df = pd.concat(kmer_maps_list, ignore_index=True)
    # search for all of the corresponding AA kmers to every PC kmer in topN for each method
    full_kmer_list: list[str] = []
    matching_kmers: list[pd.DataFrame] = []
    for kmer_list in kmer_lists:
        for each_kmer in kmer_list:
            full_kmer_list.append(each_kmer)
            matching_kmers.append(kmer_maps_df[kmer_maps_df["1"] == each_kmer]["0"])

    # make a new df holding all matching kmers with topN kmer as key
    matching_kmers_df = pd.DataFrame(
        np.nan, index=range(len(max(matching_kmers, key=len))), columns=["idx"]
    )

    for i, matching_kmer in enumerate(matching_kmers):
        matching_kmer_series = pd.Series(matching_kmer.values).reindex(
            matching_kmers_df.index
        )
        matching_kmers_df[full_kmer_list[i]] = matching_kmer_series

    # drop the place holder column 'idx' from dataframe
    matching_kmers_df.drop("idx", axis=1, inplace=True)

    # save the df of matching PC and AA kmers for each mapping method
    for mm in mapping_methods:
        matching_kmers_df.to_csv(f"topN_PC_AA_kmer_mappings_{mm}.csv", index=False)

    # if comparing more than one mapping method, find and return matches, if any
    if len(topN_files) > 1:
        # get feature columns
        kmer_features = matching_kmers_df.columns
        # find all kmer features in each column
        matching_kmers_dict = {
            kmer_feature: set(matching_kmers_df[kmer_feature].dropna())
            for kmer_feature in kmer_features
        }
        # initialize list for aggregating matching kmers
        kmer_matches = []
        # iterate through kmer feature columns to find matching kmers
        for i, kmer_feature in enumerate(kmer_features):
            for j in range(i + 1, len(kmer_features)):
                kmer_query = kmer_features[j]
                common = (
                    matching_kmers_dict[kmer_feature] & matching_kmers_dict[kmer_query]
                )
                if common:
                    kmer_matches.append(
                        {
                            mapping_methods[0]: kmer_feature,
                            mapping_methods[1]: kmer_query,
                            "matching_AA_kmers": list(common),
                        }
                    )
        kmer_matches_df = pd.DataFrame(kmer_matches)
        return kmer_matches_df
    return None
