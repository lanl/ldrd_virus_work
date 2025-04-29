import pandas as pd
from typing import Optional


def check_entries(
    surface_exposed_df: pd.DataFrame,
    check_list: list,
    df_file: str,
    reference_list: Optional[list] = None,
    explicit: bool = False,
) -> None:
    """
    check entries of the surface exposed dataframe. change surface exposure status and add references
    based on user input. Save modified dataframe.

    Parameters:
    -----------
    surface_exposed_df: pd.DataFrame
        dataframe containing virus-protein pairs and their corresponding surface exposure status with
        references where appropriate
    check_list: list
        list of virus names to check
    df_file: str
        file name for saving modified dataframe
    reference_list: list
        list of references corresponding to entries of ``check_list``
    explicit: bool
        whether or not to check values of ``check_list`` explicitly or to search for them as keywords
    """
    # check each entry in list explicitly by name
    if explicit:
        cross_reference = surface_exposed_df[
            surface_exposed_df["protein_names"].isin(check_list)
        ]
    # or check for presence of string in protein name (for checking keywords)
    else:
        all_terms = "|".join(check_list)
        cross_reference = surface_exposed_df[
            surface_exposed_df["protein_names"].str.contains(
                all_terms, case=False, na=False
            )
        ]
    # count the number of entries that do not have references
    remaining = pd.isna(cross_reference["reference"]).sum()
    for i, row in enumerate(cross_reference.itertuples()):
        # check entries without references
        if pd.isna(row.reference):
            print(
                f"Remaining: {remaining}, Row: {row.Index}, Status: {row.surface_exposed_status}, {row.virus_names}, {row.protein_names}"
            )
            # if given a reference list, skip manual entries and instead assign reference based on list
            if reference_list is not None:
                protein_name = row.protein_names
                list_idx = check_list.index(protein_name)
                reference = reference_list[list_idx]
                surface_exposed_df.iloc[row.Index].reference = reference  # type:ignore
                remaining -= 1
            else:
                reference = input("Reference: ")
                if reference == "exit":
                    break
                # manually fix entry if necessary
                elif reference == "fix":
                    new_status = input("Surface exposure status: ")
                    surface_exposed_df.iloc[row.Index].surface_exposed_status = new_status  # type: ignore
                    # give reference for change
                    reference = input("Reference: ")
                    surface_exposed_df.iloc[
                        row.Index
                    ].reference = reference  # type:ignore
                    remaining -= 1
                else:
                    surface_exposed_df.iloc[
                        row.Index
                    ].reference = reference  # type:ignore
                    remaining -= 1

    surface_exposed_df.to_csv(df_file, index=False)
    print("Saved 'surface_exposed_df.csv'.")


if __name__ == "__main__":
    purge_list = [
        "membrane protein M",
        "1B(VP2)",
        "1C(VP3)",
        "1D(VP1)",
        "Envelope surface glycoprotein gp120",
        "3C",
        "3C protein",
        "3D",
        "3D protein",
        "3D-POL protein",
        "Hel protein",
        "Lab protein",
        "Lb protein",
        "1A(VP4)",
        "nucleocapsid",
        "p1",
        "p2",
        "p6",
        "p66 subunit",
        "p7 protein",
        "pre-membrane protein prM",
        "protein VP0",
        "protein pr",
        "protien 3A",
        "protein 1A",
        "protein 1B",
        "protein 1C",
        "protein 1D",
        "protein 2A",
        "protein 2B",
        "protein 2C",
        "protien 2K",
        "protein 3A",
        "protein 3AB",
        "protein 3C",
        "protein 3D",
    ]

    references = [
        "https://doi.org/10.1099/0022-1317-69-5-1105",
        "https://doi.org/10.3389/fmicb.2020.562768",
        "https://doi.org/10.3389/fmicb.2020.562768",
        "https://doi.org/10.3389/fmicb.2020.562768",
        "https://doi.org/10.1038/31405",
        "https://doi.org/10.3390/v15122413",
        "https://doi.org/10.3390/v15122413",
        "https://doi.org/10.3389/fimmu.2024.1365521",
        "https://doi.org/10.3389/fimmu.2024.1365521",
        "https://doi.org/10.3389/fimmu.2024.1365521",
        "https://doi.org/10.1016/j.virusres.2024.199401",
        "https://doi.org/10.1128/jvi.74.24.11708-11716.2000",
        "https://doi.org/10.1128/jvi.74.24.11708-11716.2000",
        "https://doi.org/10.3389/fmicb.2020.562768",
        "https://doi.org/10.1007/s11904-011-0107-3",
        "https://doi.org/10.1007/s11904-011-0107-3",
        "https://doi.org/10.1007/s11904-011-0107-3",
        "https://doi.org/10.1007/s11904-011-0107-3",
        "https://doi.org/10.1002/cbic.202000263",
        "https://doi.org/10.1038/s41598-019-44413-x",
        "https://doi.org/10.1016/0042-6822(92)90267-S",
        "https://doi.org/10.1128/jvi.73.11.9072-9079.1999",
        "https://doi.org/10.1042/BJ20061136",
        "https://doi.org/10.1128/jvi.00791-17",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
        "https://viralzone.expasy.org/99",
    ]

    df_file = "surface_exposed_df.csv"
    surface_exposed_df = pd.read_csv("surface_exposed_df.csv")
    check_entries(surface_exposed_df, purge_list, df_file, references, explicit=True)
