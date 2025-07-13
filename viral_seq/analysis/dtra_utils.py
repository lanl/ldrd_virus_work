import pandas as pd
from typing import Optional, Literal
import numpy as np
import functools
from importlib.resources import files
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


def find_matching_kmers(
    target_column: str, mapping_methods: list[Literal["jurgen_schmidt", "shen_2007"]]
) -> str:
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
    mapping_methods: list[Literal["jurgen_schmidt", "shen_2007"]]
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
        topN_df_list = []
        for mm in mapping_methods:
            topN_df = pl.read_parquet(
                f"topN_kmers_{target_column}_{mm}.parquet.gzip"
            ).to_pandas()
            topN_df_list.append(topN_df)
        kmer_matches = match_kmers(topN_df_list, mapping_methods)
        if kmer_matches is None or kmer_matches.empty:
            return "No matching AA kmers found between PC mapping schemes in TopN."
        else:
            kmer_matches.to_csv(
                f"{mapping_methods[0]}_{mapping_methods[1]}_kmer_matches.csv",
                index=False,
            )
            print("Saved matching AA kmers in topN between PC mapping methods.")
            return f"Matching AA kmers between mapping methods:\n {kmer_matches.to_string(index=False)}"
    except FileNotFoundError:
        return "Must run workflow using both mapping methods before performing kmer mapping."


def match_kmers(
    topN_df_list: list[pd.DataFrame],
    mapping_methods: list[Literal["jurgen_schmidt", "shen_2007"]],
    save_dir: str = "kmer_maps",
    aa_codes: str = "ACDEFGHIKLMNPQRSTVWYX*",
) -> Optional[pd.DataFrame]:
    """
    determine matching AA kmer(s) between different PC mappings from topN kmer features
    using each mapping method (i.e. ``jurgen_schmidt`` and ``shen_2007``). If matches
    exist, return dataframe containing matching AA kmers and corresponding PC kmer from
    each mapping method (or explicit AA matches).

    in order for this function to execute, you must run the full workflow using both
    `--mapping-method` arguments to generate the required kmer mapping files

    Parameters:
    -----------
    topN_df_list: list[pd.DataFrame]
        list of pandas dataframes containing topN kmers found using each mapping scheme
    mapping_methods: list[Literal["jurgen_schmidt", "shen_2007"]]
        names of mapping methods to compare
    save_dir: str
        file name for saving kmer_maps
    aa_codes: str
        string containing single-letter codes for 20 natural amino acids

    Returns:
    -------
    kmer_matches_df: Optional[pd.DataFrame]
        dataframe of PC kmer matches and their associated AA-kmer(s)
    """

    # initialize list for storing PC kmer mappings to be compared for matching AA kmers
    topN_kmer_mappings_all = []
    for i, mm in enumerate(mapping_methods):
        # check that mapping method is recognized
        if mm not in ["jurgen_schmidt", "shen_2007"]:
            raise ValueError(
                "Mapping method not recognized from ``jurgen_schmidt``, ``shen_2007``."
            )
        # load appropriate kmer-matching files for each mapping method individually
        # only load ``kmer_maps`` files containing the same length kmers as in topN
        topN_mm = topN_df_list[i]
        # check that no AA kmers contain invalid characters (outside of 20 aa single letter codes)
        aa_kmer_values = set(
            "".join([k[8:] for k in topN_mm["0"] if k[:8] == "kmer_AA_"])
        )
        if not aa_kmer_values.issubset(set(aa_codes)):
            raise ValueError("AA-kmers contain incorrect values.")
        # load PC-AA kmer maps
        kmer_maps_df = pl.read_parquet(
            f"{save_dir}/kmer_maps_{mm}.parquet.gzip"
        ).to_pandas()
        # add identity map for AA_kmers in the topN
        top_AA_kmers = [
            top_kmer for top_kmer in topN_mm["0"] if top_kmer.startswith("kmer_AA_")
        ]
        if len(top_AA_kmers) > 0:
            identity_map = pd.DataFrame(
                {
                    "0": top_AA_kmers,
                    "1": top_AA_kmers,
                }
            )
            kmer_maps_df = pd.concat([kmer_maps_df, identity_map], ignore_index=True)
        # find kmer maps that match topN features
        kmer_maps_topN = kmer_maps_df[kmer_maps_df["0"].isin(topN_mm["0"])]
        topN_kmer_mappings_all.append(kmer_maps_topN)
        # reorganize kmer mappings for readability and save topN mappings
        topN_kmer_mappings = kmer_maps_topN.groupby("0")["1"].apply(list).to_frame()
        topN_kmer_mappings = pd.DataFrame(
            topN_kmer_mappings["1"].tolist(), index=topN_kmer_mappings.index
        ).T
        topN_kmer_mappings.to_csv(f"topN_PC_AA_kmer_mappings_{mm}.csv", index=False)
    # if workflow has been run with both mapping methods, match kmers between schemes
    if len(topN_kmer_mappings_all) > 1:
        kmer_matches_df = pd.merge(
            topN_kmer_mappings_all[0], topN_kmer_mappings_all[1], on="1"
        )
        if not kmer_matches_df.empty:
            kmer_matches_df.columns = [
                mapping_methods[0],
                "matching_AA_kmers",
                mapping_methods[1],
            ]  # type: ignore
            kmer_matches_df = kmer_matches_df[mapping_methods + ["matching_AA_kmers"]]
            kmer_matches_groups = kmer_matches_df.groupby(
                mapping_methods, as_index=False
            )["matching_AA_kmers"].apply(list)
            # combine PC matches with multiple AA matches into a single row with AA matches in individual columns
            expanded_matches = pd.DataFrame(
                kmer_matches_groups["matching_AA_kmers"].tolist(),
                index=kmer_matches_groups.index,
            )
            expanded_matches.columns = [
                f"matching AA kmer {i}" for i in range(expanded_matches.shape[1])
            ]  # type: ignore
            kmer_matches_out = pd.concat(
                [kmer_matches_groups, expanded_matches], axis=1
            ).drop(columns="matching_AA_kmers")
            return kmer_matches_out
    return None
