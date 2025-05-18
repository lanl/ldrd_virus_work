import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import itertools
from typing import Literal, Optional


def pc_to_aa(
    kmer_peptide: str, mapping_method: Literal["jurgen_schmidt", "shen_2007"]
) -> list[str]:
    """
    translate PC to AA kmers using the provided mapping method

    Parameters:
    -----------
    kmer_peptide: str
        PC-kmer peptide to translate. input should be just
        a string of numbers without ``kmer_PC_`` prefix
    mapping_method: Literal["jurgen_schmidt", "shen_2007"]
        mapping scheme for translating PC to AA kmers

    Returns:
    --------
    aa_seq_list: list
        list of AA sequences translated from PC kmers
    """
    # TODO: currently there is no support for 'J' in hydrophobicity score translations (see issue #121)
    translations = {
        "shen_2007": {
            "1": ["A", "G", "V"],
            "2": ["C"],
            "3": ["F", "L", "I", "P"],
            "4": ["M", "S", "T", "Y"],
            "5": ["H", "N", "Q", "W"],
            "6": ["D", "E"],
            "7": ["K", "R"],
        },
        "jurgen_schmidt": {
            "0": ["A", "G"],
            "1": ["C"],
            "2": ["S", "T", "Y"],
            "3": ["N", "H", "Q"],
            "4": ["D", "E"],
            "5": ["K", "R"],
            "6": ["I", "V", "L", "M"],
            "7": ["F", "W"],
            "8": ["P"],
        },
    }
    # get lists of AA mappings from each value in PC kmer
    kmer_maps = [translations[mapping_method][s] for s in kmer_peptide]
    # create exhaustive lists of all AA kmer mappings
    aa_seq = itertools.product(*kmer_maps)
    # join values to make AA kmer mappings
    aa_seq_list = ["".join(s) for s in aa_seq]
    return aa_seq_list


def hydrophobicity_score(
    kmer_list: list[str],
    mapping_method: Optional[Literal["jurgen_schmidt", "shen_2007"]] = None,
) -> pd.DataFrame:
    """
    calculate the hydrophobicity score of input kmers using
    Grand Average of Hydropathy (GRAVY, https://doi.org/10.1016/0022-2836(82)90515-0)

    Parameters:
    -----------
    kmer_list: list[str]
        list containing topN kmers including prefixes (i.e. ``kmer_AA_`` or ``kmer_PC_``)
    mapping_method: Optional[Literal["jurgen_schmidt", "shen_2007"]]
        mapping method for determining AA kmers associated with PC kmer inputs

    Returns:
    --------
    kmer_gravy_df: pd.DataFrame
        pandas dataframe containing kmer inputs, translated peptides and hydrophobicity scores
    """

    kmer_gravy = []
    # iterate through kmers
    for kmer in kmer_list:
        # determine status of kmer as PC or AA
        if kmer[:8] == "kmer_AA_":
            aa_peptides = [kmer[8:]]
        else:
            # if PC kmer, find all AA kmer maps corresponding to PC kmer
            pc_peptide = kmer[8:]
            # TODO: determine whether translating pc to aa kmers will be necessary
            # after merge of !115 (tracing kmers) vs. using explicit matches from dataset
            if mapping_method is not None:
                aa_peptides = pc_to_aa(pc_peptide, mapping_method)
            # check that if PC kmers in list, a mapping method is provided
            else:
                raise ValueError(
                    "Please provide a mapping method ('jurgen_schmidt' or 'shen_2007') for performing PC-AA translations."
                )

        # calculate hydrophobicity scores for all kmers
        for aa_peptide in aa_peptides:
            protein_info = ProteinAnalysis(aa_peptide)
            kmer_gravy.append((kmer, aa_peptide, protein_info.gravy()))

    kmer_gravy_df = pd.DataFrame.from_records(kmer_gravy).rename(
        columns={0: "kmer", 1: "peptide", 2: "score"}
    )
    # reverse order of gravy scores to match FIC plot ranking
    kmer_gravy_df = kmer_gravy_df[::-1].reset_index(drop=True)
    return kmer_gravy_df
