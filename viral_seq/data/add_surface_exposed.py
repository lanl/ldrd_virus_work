import pandas as pd


def add_surface_exposed(surface_exposed_df: pd.DataFrame, surface_exposed_list: list):
    """
    add "surface_exposure_status" values to entries in dataframe containing
    virus and protein names from dataset and save dataframe with new entries
    """
    response_list = {}
    # non-exhaustive list of non-structural protein names, parts of names (i.e. 'ase' as in protease)
    not_exposed = [
        "ase",
        "nonstructural",
        "RNA",
        "DNA",
        "polyprotein",
        "NS",
        "nsp",
        "Hel",
        "Pol",
        "pol",
        "3C",
        "hypothetical",
        "pTP",
        "TP",
        "large",
        "ORF",
        "100k",
    ]
    exposed = [
        "membrane",
        "glycoprotein",
        "structural",
        "envelope",
        "III",
        "spike",
        "matrix",
    ]
    for i, row in enumerate(surface_exposed_df.itertuples()):
        if pd.isna(row.surface_exposed_status):
            if any(s in row.protein_names for s in not_exposed):
                response_list[i] = "no"
                continue
            if any(s in row.protein_names for s in exposed) or any(
                s in row.protein_names for s in surface_exposed_list
            ):
                response_list[i] = "yes"
                continue
            else:
                print(f"{row.Index}, {row.virus_names}, {row.protein_names}")
                response = input("surface exposure status:")
                if response == "exit":
                    surface_exposed_df.loc[
                        list(response_list.keys()), "surface_exposed_status"
                    ] = list(response_list.values())
                    surface_exposed_df.to_csv(df_file, index=False)
                    break
                else:
                    response_list[i] = response


surface_exposed_list = [
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
    "Asp",  # HIV-1 Antisense Protein
    "CR1-beta",
]

df_file = "surface_exposed_df.csv"
surface_exposed_df = pd.read_csv("surface_exposed_df.csv")
add_surface_exposed(surface_exposed_df, surface_exposed_list)
