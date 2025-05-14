import pandas as pd
from typing import Optional
from importlib.resources import files


def add_surface_exposed(
    surface_exposed_df: pd.DataFrame,
    save_file: str,
    check_entries: Optional[list[str]] = None,
    explicit: bool = False,
) -> None:
    """
    add "surface_exposure_status" values (yes/no) and corresponding references
    to dataframe containing virus-protein pairs and save modified dataframe

    surface exposure status decisions are determined either programmatically by checking
    against lists of known (not) surface exposed proteins/keywords or manually by
    taking user input for individual entries one at a time where a decision is not already present

    providing a list of string values representing keywords or specific proteins in the dataset
    to ``check_entries`` allows the user to double-check entries either explicitly matching the
    strings (``explicit=True``) or containing a given keyword (``explicit=False``)

    Parameters:
    -----------
    surface_exposed_df: pd.DataFrame
        dataframe containing corresponding virus-protein pairs for which to add surface exposure status
        and reference for decision if not already present
    save_file: str
        file name for saving modified dataframe as csv
    check_entries: Optional[list[str]]
        flag for checking/changing specific entries. this option provides the capability to change
        viral entries based on cross-reference of the surface_exposed_df ``protein_names`` against
        an input list of values using either keyword search (with ``explicit=False``) or explicit
        string search (with ``explicit=True``)
    explicit: bool
        whether or not to check entries explicitly or to search using keywords when using ``check_entries``
    """

    response_dict = {}
    reference_dict = {}
    # non-exhaustive list of non-structural protein names, parts of names (i.e. 'ase' as in protease)
    not_exposed: list[str] = [
        "ase",  # https://doi.org/10.1016/j.sbi.2006.10.010
        "nonstructural",  # https://meshb.nlm.nih.gov/record/ui?name=Viral+Nonstructural+Proteins
        "RNA",  # related to RNA polymerase,
        "DNA",  # related to DNA polymerase
        "polyprotein",  # viral polyproteins are proteolytically cleaved during the viral maturation cycle and may contain
        # mature products that are surface exposed (i.e. prM) but are not considered "surface-exposed" proteins https://doi.org/10.1016/j.coviro.2013.03.009
        "Hel",  # abbreviation of ``helicase`` https://doi.org/10.1016/S0959-440X(02)00298-1
        "3C",  # refers to ``3C proteinase`` of picornaviridae https://doi.org/10.3390/v8030082
        "hypothetical",  # proteins for which the function is  unknown. https://doi.org/10.1002/cfg.66
        "pTP",  # precursor terminal protein, involved in viral replication https://doi.org/10.1128/jvi.69.7.4079-4085.1995
        "100k",  # https://www.uniprot.org/uniprotkb/P24932/entry
        "IVa2",  # https://www.uniprot.org/uniprotkb/P03272/entry
        "tegument",  # the proteins of herpesviruses that lie between the capsid and viral envelope https://doi.org/10.1128/mmbr.00040-07
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
    not_exposed_exceptions: list[str] = [
        "hemagglutinin-neuraminidase",  # surface exposed protein on paramyxoviridae https://viralzone.expasy.org/556
        "hemagglutinin-esterase",  # surface exposed protein of influenza (https://doi.org/10.1007/s13238-015-0193-x), coronavirus (https://doi.org/10.1007/978-1-4899-1531-3_8), and torovirus (https://doi.org/10.1128/jvi.71.7.5277-5286.1997)
        "neuraminidase",  # surface exposed protein of Influenza A (https://doi.org/10.3389/fmicb.2019.00039)
    ]
    exposed: list[str] = [
        "glycoprotein",  # surface proteins of enveloped viruses https://doi.org/10.1093/clinids/2.1.40
        "envelope",  # referring to proteins associated with the viral envelope https://www.uniprot.org/keywords/KW-0261
        "spike",  # referring to coronavirus spike proteins https://doi.org/10.1146/annurev-virology-110615-042301
        "hemagglutinin-esterase",  # surface exposed protein of influenza (https://doi.org/10.1007/s13238-015-0193-x), coronavirus (https://doi.org/10.1007/978-1-4899-1531-3_8), and torovirus (https://doi.org/10.1128/jvi.71.7.5277-5286.1997)
        "hemagglutinin-neuraminidase",  # surface exposed protein on paramyxoviridae https://viralzone.expasy.org/556
        "fusion",  # associated with membrane proteins that facilitate fusion of viral envelopes https://doi.org/10.1038/nsmb.1456
        "hemagglutinin",  # viral fusion protein of influenza https://doi.org/10.1038/nsmb.1456
        "fiber",  # referring to the viral surface protein of adenoviridae https://viralzone.expasy.org/183
        "neuraminidase",  # surface exposed protein of Influenza A (https://doi.org/10.3389/fmicb.2019.00039)
        "G1",  # glycoprotein on the surface of hanntaviridae https://viralzone.expasy.org/7079
        "G2",  # glycoprotein on the surface of hanntaviridae https://viralzone.expasy.org/7079
    ]
    # cross-reference dataframe entries if provided
    # a list of protein keywords/names to check
    if check_entries:
        check_column = "protein_names"
        # either check explicitly if they exist in the protein names
        if explicit:
            cross_reference = surface_exposed_df[
                surface_exposed_df[check_column].isin(check_entries)
            ]
        # or search for them as keywords in each string
        else:
            all_terms = "|".join(check_entries)
            cross_reference = surface_exposed_df[
                surface_exposed_df[check_column].str.contains(
                    all_terms, case=False, na=False
                )
            ]
        remaining = len(cross_reference)
    else:
        cross_reference = surface_exposed_df
        remaining = surface_exposed_df["surface_exposed_status"].isna().sum()
    for i, row in enumerate(cross_reference.itertuples()):
        # if there is no surface exposure status or if ``check_entries``
        # is not an empty list, perform labeling
        if pd.isna(row.surface_exposed_status) or check_entries:
            if isinstance(row.protein_names, str):
                protein_query: str = row.protein_names.lower()
            else:
                raise TypeError("Invalid protein query type: expected 'str' value.")
            # only perform programmatic labeling if not checking entries
            if not check_entries:
                if any(s.lower() in protein_query for s in not_exposed) and not any(
                    s.lower() in protein_query for s in not_exposed_exceptions
                ):
                    response_dict[i] = "no"
                    reference_dict[
                        i
                    ] = "labeling performed programmatically using 'not_exposed' list"
                    remaining -= 1
                    continue
                if any(s.lower() in protein_query for s in exposed):
                    response_dict[i] = "yes"
                    reference_dict[
                        i
                    ] = "labeling performed programmatically using 'exposed' list"
                    remaining -= 1
                    continue
            print(
                f"{row.Index}, ({remaining}), Virus Name: {row.virus_names}, Protein Name: {row.protein_names}, Status: {row.surface_exposed_status}, Reference: {row.reference}"
            )
            response_1 = input("surface exposure status:")
            # if empty response, skip entry
            if response_1 == "":
                continue
            if response_1 == "fix":
                print("Enter new surface exposure status and reference...")
                new_status = input("Surface exposure status: ")
                surface_exposed_df.iloc[row.Index].surface_exposed_status = new_status  # type: ignore
                # give reference for change
                reference = input("Reference: ")
                surface_exposed_df.iloc[row.Index].reference = reference  # type:ignore
                remaining -= 1
                continue
            elif response_1 == "exit":
                surface_exposed_df.loc[
                    list(response_dict.keys()), "surface_exposed_status"
                ] = list(response_dict.values())
                surface_exposed_df.loc[list(reference_dict.keys()), "reference"] = list(
                    reference_dict.values()
                )
                surface_exposed_df.to_csv(save_file, index=False)
                break
            else:
                response_2 = input("reference:")
                if not response_2:
                    response_2 = "None"
                response_dict[i] = response_1
                reference_dict[i] = response_2
            remaining -= 1
    # save responses after finishing
    surface_exposed_df.loc[list(response_dict.keys()), "surface_exposed_status"] = list(
        response_dict.values()
    )
    surface_exposed_df.loc[list(reference_dict.keys()), "reference"] = list(
        reference_dict.values()
    )
    surface_exposed_df.to_csv(save_file, index=False)


if __name__ == "__main__":
    df_file = "surface_exposed_df.csv"
    df_path = files("viral_seq.data") / df_file
    surface_exposed_df = pd.read_csv(df_path)  # type: ignore
    add_surface_exposed(surface_exposed_df, save_file=df_file)
