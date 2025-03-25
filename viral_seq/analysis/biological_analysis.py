import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import itertools
from collections import defaultdict


def pc_to_aa(kmer_peptide: str, mapping_method: str) -> list:
    """
    translate PC to AA kmers using the provided mapping method
    TODO: currently there is no support for 'J' in hydrophobicity score translations

    Parameters:
    -----------
        kmer_peptide: str
                PC kmer_peptide to translate
        mapping_method: str
                mapping scheme for translating PC to AA kmers

        Returns:
        --------
        aa_seq_list: list
                list of AA sequences translated rom PC kmers
    """
    translations = {
        "shen_2007": {
            "A": ["A", "G", "V"],
            "B": ["C"],
            "C": ["F", "L", "I", "P"],
            "D": ["M", "S", "T", "Y"],
            "E": ["H", "N", "Q", "W"],
            "F": ["D", "E"],
            "G": ["K", "R"],
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


def hydrophobicity_score(kmer_list: list, mapping_method: str) -> pd.DataFrame:
    """
    calculate the hydrophobicity score of input kmers using
    Grand Average of Hydropathy (GRAVY, https://doi.org/10.1016/0022-2836(82)90515-0)

    Parameters:
    -----------
    kmer_list: list
            list containing topN kmers
    mapping_method: str
            mapping method for determing AA kmers associated with PC kmer inputs

    Returns:
    --------
    kmer_gravy_df: pd.DataFrame
            dictionary containing kmer inputs, translated peptides and hydrophobicity scores
    """
    kmer_gravy = defaultdict(list)
    # iterate through kmers
    for kmer in kmer_list:
        # determine status of kmer as PC or AA
        if kmer[:8] == "kmer_AA_":
            kmer_status = "AA"
        else:
            kmer_status = "PC"

        # if PC kmer, find all AA kmer maps corresponding to PC kmer
        if kmer_status == "PC":
            pc_peptide = kmer[8:]
            # TODO: translating pc to aa kmers will be unnecessary after merge of !115 (tracing kmers)
            aa_peptides = pc_to_aa(pc_peptide, mapping_method)
        else:
            aa_peptides = [kmer[8:]]

        # calculate hydrophobicity scores for all kmers
        for aa_peptide in aa_peptides:
            protein_info = ProteinAnalysis(aa_peptide)
            kmer_gravy[kmer].append((aa_peptide, protein_info.gravy()))

    kmer_gravy_df = pd.DataFrame.from_dict(kmer_gravy, orient="index")
    return kmer_gravy_df
