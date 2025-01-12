import pandas as pd


def add_surface_exposed(surface_exposed_df: pd.DataFrame, surface_exposed_list: list):
    """
    add "surface_exposure_status" values to entries in dataframe containing
    virus and protein names from dataset and save dataframe with new entries
    """
    response_list = {}
    reference_list = {}
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
        "ORF",
        "100k",
        "IVa2",
    ]
    not_exposed_exceptions = ["hemagglutinin-neuraminidase", "hemagglutinin-esterase"]
    exposed = [
        "membrane",
        "glycoprotein",
        "structural",
        "envelope",
        "III",
        "spike",
    ]
    for i, row in enumerate(surface_exposed_df.itertuples()):
        if pd.isna(row.surface_exposed_status):
            if any(s in row.protein_names for s in not_exposed) and not any(
                s in row.protein_names for s in not_exposed_exceptions
            ):
                response_list[i] = "no"
                reference_list[i] = "None"
                continue
            if any(s in row.protein_names for s in exposed) or any(
                s in row.protein_names for s in surface_exposed_list
            ):
                response_list[i] = "yes"
                reference_list[i] = "None"
                continue
            else:
                print(f"{row.Index}, {row.virus_names}, {row.protein_names}")
                response_1 = input("surface exposure status:")
                if response_1 == "exit":
                    surface_exposed_df.loc[
                        list(response_list.keys()), "surface_exposed_status"
                    ] = list(response_list.values())
                    surface_exposed_df.loc[
                        list(reference_list.keys()), "reference"
                    ] = list(reference_list.values())
                    surface_exposed_df.to_csv(df_file, index=False)
                    break
                else:
                    response_2 = input("reference:")
                    response_list[i] = response_1
                    reference_list[i] = response_2


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

# TODO: add these references to dataframe
references = [
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
urls = [
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
add_surface_exposed(surface_exposed_df, surface_exposed_list)
