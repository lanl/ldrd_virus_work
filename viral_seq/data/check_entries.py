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
    purge_list = ["ORF"]

    df_file = "surface_exposed_df.csv"
    surface_exposed_df = pd.read_csv("surface_exposed_df.csv")
    check_entries(surface_exposed_df, purge_list, df_file)
