import pandas as pd


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
