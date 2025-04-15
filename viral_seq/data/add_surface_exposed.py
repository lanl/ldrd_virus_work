import pandas as pd
from typing import List


def add_surface_exposed(surface_exposed_df: pd.DataFrame, save_file: str) -> None:
    """
    add "surface_exposure_status" values to entries in dataframe containing
    virus and protein names from dataset and save dataframe with new entries

    Parameters:
    -----------
    surface_exposed_df: pd.DataFrame
        dataframe containing corresponding virus-protein pairs as well as surface exposed (yes/no) status
        and reference for decision if not determined from known lists of (not) surface-exposed proteins
    save_file: str
        file name for saving modified dataframe as csv
    """

    response_list = {}
    reference_list = {}
    # non-exhaustive list of non-structural protein names, parts of names (i.e. 'ase' as in protease)
    not_exposed: List[str] = [
        "ase",  # https://doi.org/10.1016/j.sbi.2006.10.010
        "nonstructural",  # https://meshb.nlm.nih.gov/record/ui?name=Viral+Nonstructural+Proteins
        "RNA",  # related to RNA polymerase,
        "DNA",  # related to DNA polymerase
        "polyprotein",  # viral polyproteins are proteolytically cleaved during the viral maturation cycle and may contain
        # mature products that are surface exposed (i.e. prM) but are not considered "surface-exposed" proteins https://doi.org/10.1016/j.coviro.2013.03.009
        "NS",  # acronym meaning ``non-structural``
        "nsp",  # acronym for ``non-structural protein``
        "Hel",  # abbreviation of ``helicase`` https://doi.org/10.1016/S0959-440X(02)00298-1
        "pol",  # abbreviation of ``Polymerase`` https://doi.org/10.1007/978-1-4614-0980-9_12
        "3C",  # refers to ``3C proteinase`` of picornaviridae https://doi.org/10.3390/v8030082
        "hypothetical",  # proteins for which the function is  unknown. https://doi.org/10.1002/cfg.66
        "pTP",  # precursor terminal protein, involved in viral replication https://doi.org/10.1128/jvi.69.7.4079-4085.1995
        "TP",  # terminal protein, involved in viral replication https://doi.org/10.1128/jvi.69.7.4079-4085.1995
        "100k",  # https://www.uniprot.org/uniprotkb/P24932/entry
        "IVa2",  # https://www.uniprot.org/uniprotkb/P03272/entry
        "tegument",  # the proteins of herpesviruses that lie between the capsid and viral envelope https://doi.org/10.1128/mmbr.00040-07
        "nsP",  # aka non-structural protein
        "nuclear",  # associating with the host nucleus
        "nucleocapsid",  # referring to the capsid of enveloped viruses, specifically HIV-1 https://viralzone.expasy.org/7
        "transcription",  # relating to ``transcription factor`` or ``transcription regulator`` https://doi.org/10.1016/j.cell.2020.06.023
        "non-structural",  # i.e. ``non-structural protein``
        "nucleo",  # ``ribonucleotide`` or ``nucleocapsid``
        "core",  # https://www.ncbi.nlm.nih.gov/mesh?Db=mesh&Cmd=DetailsSearch&Term=%22Viral+Core+Proteins%22%5BMeSH+Terms%5D
        "regulator",  # https://www.ncbi.nlm.nih.gov/mesh?Db=mesh&Cmd=DetailsSearch&Term=%22Viral+Regulatory+and+Accessory+Proteins%22%5BMeSH+Terms%5D
        "gag",  # associated with late stage viral assembly https://viralzone.expasy.org/5068
        "small t antigen",  # referring to regulatory proteins of polyomaviruses https://doi.org/10.3390/ijms20163914
        "large t antigen",  # referring to regulatory proteins of polyomaviruses https://doi.org/10.3390/ijms20163914
    ]
    not_exposed_exceptions: List[str] = [
        "hemagglutinin-neuraminidase",
        "hemagglutinin-esterase",
        "neuraminidase",
    ]
    exposed: List[str] = [
        "membrane",
        "glycoprotein",
        "structural",
        "envelope",
        "III",
        "spike",
        "hemagglutinin-esterase",
        "hemagglutinin-neuraminidase",
        "fusion",
        "hemagglutinin",
        "fiber",
        "HA",
        "NA",
        "neuraminidase",
        "G1",
        "G2",
    ]
    remaining = surface_exposed_df["surface_exposed_status"].isna().sum()
    for i, row in enumerate(surface_exposed_df.itertuples()):
        if pd.isna(row.surface_exposed_status):
            if isinstance(row.protein_names, str):
                protein_query: str = row.protein_names.lower()
            else:
                raise TypeError("Invalid protein query type: expected 'str' value.")
            if any(s.lower() in protein_query for s in not_exposed) and not any(
                s.lower() in protein_query for s in not_exposed_exceptions
            ):
                response_list[i] = "no"
                reference_list[i] = "None"
                remaining -= 1
                continue
            if any(s.lower() in protein_query for s in exposed):
                response_list[i] = "yes"
                reference_list[i] = "None"
                remaining -= 1
                continue
            else:
                print(f"{row.Index}, ({remaining}), {row.virus_names}, {protein_query}")
                response_1 = input("surface exposure status:")
                if response_1 == "exit":
                    surface_exposed_df.loc[
                        list(response_list.keys()), "surface_exposed_status"
                    ] = list(response_list.values())
                    surface_exposed_df.loc[
                        list(reference_list.keys()), "reference"
                    ] = list(reference_list.values())
                    surface_exposed_df.to_csv(save_file, index=False)
                    break
                else:
                    response_2 = input("reference:")
                    if not response_2:
                        response_2 = "None"
                    response_list[i] = response_1
                    reference_list[i] = response_2
                remaining -= 1
    # save responses after finishing
    surface_exposed_df.loc[list(response_list.keys()), "surface_exposed_status"] = list(
        response_list.values()
    )
    surface_exposed_df.loc[list(reference_list.keys()), "reference"] = list(
        reference_list.values()
    )
    surface_exposed_df.to_csv(save_file, index=False)


if __name__ == "__main__":
    # TODO: these values (i.e. "references", "urls") are left over previous surface exposed list, consider purging
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
    add_surface_exposed(surface_exposed_df, save_file=df_file)
