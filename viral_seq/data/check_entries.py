import pandas as pd


def check_entries(
    surface_exposed_df: pd.DataFrame, check_list: list, df_file: str
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
    """
    cross_reference = surface_exposed_df[
        surface_exposed_df["protein_names"].isin(check_list)
    ]
    # count the number of entries that do not have references
    remaining = pd.isna(cross_reference["reference"]).sum()
    for i, row in enumerate(cross_reference.itertuples()):
        # check entries without references
        if pd.isna(row.reference):
            print(
                f"Remaining: {remaining}, Row: {row.Index}, {row.virus_names}, {row.protein_names}"
            )
            reference = input("Reference: ")
            if reference == "exit":
                break
            # manually fix entry if necessary
            elif reference == "fix":
                new_status = input("Surface exposure status: ")
                surface_exposed_df.iloc[row.Index].surface_exposed_status = new_status  # type: ignore
                # give reference for change
                reference = input("Reference: ")
                surface_exposed_df.iloc[row.Index].reference = reference  # type:ignore
                remaining -= 1
            else:
                surface_exposed_df.iloc[row.Index].reference = reference  # type:ignore
                remaining -= 1

    surface_exposed_df.to_csv(df_file, index=False)
    print("Saved 'surface_exposed_df.csv'.")


if __name__ == "__main__":
    purge_list = [
        "1B(VP2)",
        "1C(VP3)",
        "1D(VP1)",
        "Envelope surface glycoprotein gp120",
        "Envelope surface glycoprotein gp160, precursor",
        "PreM protein",
        "VP1",
        "VP1 protein",
        "VP2",
        "VP2 protein",
        "VP3",
        "VP3 protein",
        "envelope glycoprotein E1",
        "envelope glycoprotein E2",
        "envelope protein",
        "envelope protein E",
        "membrane glycoprotein M",
        "membrane glycoprotein precursor prM",
        "membrane protein M",
        "hemagglutinin-neuraminidase",
        "envelope glycoprotein 150",
        "envelope glycoprotein B",
        "envelope glycoprotein E",
        "envelope glycoprotein G",
        "envelope glycoprotein H",
        "envelope glycoprotein M",
        "envelope glycoprotein UL37",
        "envelope protein",
        "envelope protein E",
        "envelope protein UL20",
        "envelope protein UL43",
        "membrane glycoprotein",
        "membrane glycoprotein UL16",
        "membrane glycoprotein UL40",
        "membrane protein UL120",
        "membrane protein UL124",
        "membrane protein UL20",
        "membrane protein UL45",
        "membrane protein UL56",
        "membrane protein US12",
        "membrane protein US15",
        "membrane protein US30",
        "membrane protein V1",
        "hexon",
        "hexon protein",
        "3A",
        "3A protein",
        "Asp",  # HIV-1 Antisense Protein, https://doi.org/10.1128/jvi.00574-19
        "CR1-beta",
    ]

    df_file = "surface_exposed_df.csv"
    surface_exposed_df = pd.read_csv("surface_exposed_df.csv")
    import pdb

    pdb.set_trace()
    check_entries(surface_exposed_df, purge_list, df_file)
