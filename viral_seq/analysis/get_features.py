from functools import cache
from collections import defaultdict
from Bio.Data.CodonTable import standard_dna_table
from Bio.SeqUtils import gc_fraction
from skbio import Sequence
from typing import Any
import pandas as pd
import numpy as np
import scipy.stats

codontab = standard_dna_table.forward_table.copy()
for codon in standard_dna_table.stop_codons:
    codontab[codon] = "STOP"


def get_similarity_features(
    df_similarity: pd.DataFrame, df_features: pd.DataFrame, suffix="_sim"
):
    hist_dists = np.array([])
    hists = (
        df_similarity.where(df_similarity != 0)
        .apply(
            lambda x: np.histogram(x[~np.isnan(x)], bins="auto", density=True), axis=0
        )
        .values
    )
    hist_dists = np.apply_along_axis(
        scipy.stats.rv_histogram, axis=0, arr=hists, density=True
    )
    simfeats: dict[str, Any] = {}
    for i, feature in enumerate(df_similarity.columns):
        if feature in df_features:
            simfeats[feature] = df_features[feature].apply(hist_dists[i].pdf)
    df_simfeats = pd.DataFrame.from_dict(simfeats)
    return df_features.join(df_simfeats, rsuffix=suffix)


def get_kmers(records, k=10, kmer_type="AA"):
    kmers = defaultdict(int)
    for record in records:
        for feature in record.features:
            if feature.type == "CDS":
                nuc_seq = feature.location.extract(record.seq)
                if len(nuc_seq) % 3 != 0:
                    # bad cds are skipped as in https://github.com/Nardus/zoonotic_rank/blob/main/Utils/GenomeFeatures.py#L105
                    continue
                this_seq = nuc_seq.translate()
                if kmer_type == "PC":
                    new_seq = ""
                    for each in this_seq:
                        new_seq += aa_map(each, method="shen_2007")
                    this_seq = new_seq
                for kmer in Sequence(str(this_seq)).iter_kmers(k, overlap=True):
                    kmers["kmer_" + kmer_type + "_" + str(kmer)] += 1
    return kmers


def get_gc(records):
    full_sequence = ""
    for record in records:
        full_sequence += str(record.seq)
    return {"GC Content": gc_fraction(full_sequence)}


# Calculates all genomic features for a given record
def get_genomic_features(records):
    coding_pairs = []
    bridge_pairs = []
    nonbridge_pairs = []
    all_cnt_dict = defaultdict(float)
    coding_cnt_dict = defaultdict(float)
    bridge_cnt_dict = defaultdict(float)
    codons = []
    full_sequence = ""
    for record in records:
        full_sequence += str(record.seq)
        # get cds for each protein so we can count bias
        for feature in record.features:
            if feature.type == "CDS":
                # need to use extract because location can be complicated
                this_seq = feature.location.extract(record.seq)
                # add pairs to lists
                try:
                    (
                        these_coding_pairs,
                        these_bridge_pairs,
                        these_nonbridge_pairs,
                        these_codons,
                    ) = split_seq(str(this_seq), coding=True)
                except AssertionError:
                    # bad cds are skipped as in https://github.com/Nardus/zoonotic_rank/blob/main/Utils/GenomeFeatures.py#L105
                    continue
                coding_pairs = coding_pairs + these_coding_pairs
                bridge_pairs = bridge_pairs + these_bridge_pairs
                nonbridge_pairs = nonbridge_pairs + these_nonbridge_pairs
                codons = codons + these_codons
                # store information about these sequences need for calculating bias
                coding_cnt_dict = get_cnt_dict(str(this_seq), coding_cnt_dict)
                bridge_cnt_dict = get_cnt_dict(
                    "".join(these_bridge_pairs), bridge_cnt_dict
                )
    if len(codons) == 0:
        print("No CDSs in features for accession", record.id)
    all_pairs = split_seq(full_sequence, coding=False)[0]
    all_cnt_dict = get_cnt_dict(full_sequence)
    # get bias for each category
    coding_ret = get_dinucleotide_bias(
        coding_pairs, coding_cnt_dict, key_prefix="coding_"
    )
    all_ret = get_dinucleotide_bias(all_pairs, all_cnt_dict, key_prefix="entire_")
    bridge_ret = get_dinucleotide_bias(
        bridge_pairs, bridge_cnt_dict, key_prefix="bridge_"
    )
    # nonbridge pairs cover entire coding sequence, so it uses same counts as coding
    nonbridge_ret = get_dinucleotide_bias(
        nonbridge_pairs, coding_cnt_dict, key_prefix="nonbridge_"
    )
    codon_ret = get_codon_amino_bias(codons)
    return {
        **coding_ret,
        **all_ret,
        **bridge_ret,
        **nonbridge_ret,
        **codon_ret,
    }


# Keeps a running count of the sequence length and number of A,T,G,C
def get_cnt_dict(seq: str, cnt_dict=None):
    if cnt_dict is None:
        cnt_dict = defaultdict(float)
    cnt_dict["seq_len"] = cnt_dict["seq_len"] + len(seq)
    for let in "ATGC":
        cnt_dict[let] = cnt_dict[let] + seq.count(let)
    return cnt_dict


# Pair list should include all pairs for this calculation
# Eg. if this is calculating dinucleotide bias for
# bridge pairs, it should include all bridge pairs for all
# coding regions for this record
# Calculation as defined in doi: 10.1126/science.aap9072
def get_dinucleotide_bias(pair_lst, cnt_dict, key_prefix=""):
    dinuc_dict = defaultdict(float)
    num_char = float(cnt_dict["seq_len"])
    num_pairs = float(len(pair_lst))
    if num_char == 0 or num_pairs == 0:
        return dinuc_dict
    # we can make the multiplicative factor to save time
    dinuc_factor = defaultdict(float)
    ATGC = ["A", "T", "G", "C"]
    for a in ATGC:
        for b in ATGC:
            dinuc_factor[a + b] = (num_char * num_char) / (
                cnt_dict[a] * cnt_dict[b] * num_pairs
            )
    # store bias value as calculated
    for pair in pair_lst:
        this_key = key_prefix + pair
        if pair not in dinuc_factor:
            continue
        this_factor = dinuc_factor[pair]
        dinuc_dict[this_key] = dinuc_dict[this_key] + this_factor
    return dinuc_dict


# Calculation as defined in doi: 10.1126/science.aap9072
def get_codon_amino_bias(codons):
    codon_count = defaultdict(float)
    amino_count = defaultdict(float)
    total_amino = len(codons)  # This includes stop codons according to reference
    bias_ret = {}
    for codon in codons:
        if codon not in codontab:
            continue
        # non-redundant codons should not be features
        if codon not in ["ATG", "TGG"]:
            codon_count[codon] = codon_count[codon] + 1.0
        amino = codontab[codon]
        amino_count[amino] = amino_count[amino] + 1.0
    for codon, count in codon_count.items():
        amino = codontab[codon]
        bias_ret[codon] = count / amino_count[amino]
    for amino, count in amino_count.items():
        if amino == "STOP":
            continue
        bias_ret[amino] = count / total_amino
    return bias_ret


# Extract pairs, if coding sequence also extract
# bridge pairs, non-bridge pairs, & codons
def split_seq(seq: str, coding: bool = False):
    seq_len = len(seq)
    bridge_pairs = []
    nonbridge_pairs = []
    codons = []
    if coding:
        # we need to have full codons
        assert seq_len % 3 == 0
        onepos_pairs = [seq[0 + i : 0 + i + 2] for i in range(0, seq_len - 1, 3)]
        twopos_pairs = [seq[1 + i : 1 + i + 2] for i in range(0, seq_len - 1, 3)]
        nonbridge_pairs = onepos_pairs + twopos_pairs
        # seq_len-3 to prevent unpaired final character
        bridge_pairs = [seq[2 + i : 2 + i + 2] for i in range(0, seq_len - 3, 3)]
        # codons
        codons = [seq[i : i + 3] for i in range(0, seq_len, 3)]
    pairs = [seq[i : i + 2] for i in range(0, seq_len - 1)]
    return pairs, bridge_pairs, nonbridge_pairs, codons


@cache
def aa_map(explicit_aa: str, *, method: str = "shen_2007") -> str:
    """
    Remap canonical single-character amino acid representations
    to other representation schemes.

    Parameters
    ----------
    explicit_aa : str
        A single-character string representing the input/explicit
        or canonical amino acid letter.
    method : str, optional
        A string representing the name of the method used to remap
        the input single-character amino acid code to some other
        amino acid representation scheme. Default is `shen_2007`.

    Returns
    -------
    mapped_aa : str
        A string representing a single character that is the remapped
        version of the input amino acid letter code.

    """
    if len(explicit_aa) != 1:
        raise ValueError(f"{explicit_aa=}; does not have length 1")
    if method == "shen_2007":
        # Categories defined in https://doi.org/10.1073/pnas.0607879104
        if explicit_aa in "AGV":
            return "A"
        elif explicit_aa in "C":
            return "B"
        elif explicit_aa in "FLIP":
            return "C"
        elif explicit_aa in "MSTY":
            return "D"
        elif explicit_aa in "HNQW":
            return "E"
        elif explicit_aa in "DE":
            return "F"
        elif explicit_aa in "KR":
            return "G"
        else:
            return "*"
    elif method == "schein_2012":
        # NOTE: these mappings appear to be inspired by
        # this manuscript from Jurgen Schmidt's collaborators
        # at UTMB: https://doi.org/10.1186/1471-2105-13-s13-s9
        # However, as far as I can tell, that manuscript uses
        # more sophisticated (eigenvector?) approaches to achieve the actual
        # mappings, so these are mostly conformant with Jurgen's
        # specific requests at
        # https://gitlab.lanl.gov/treddy/ldrd_virus_work/-/issues/67#note_307079
        if explicit_aa in "AG":
            return "0"
        elif explicit_aa in "C":
            return "1"
        elif explicit_aa in "STY":
            return "2"
        elif explicit_aa in "NQH":
            return "3"
        elif explicit_aa in "DE":
            return "4"
        elif explicit_aa in "KR":
            return "5"
        elif explicit_aa in "IVLM":
            # NOTE: Jurgen labeled this "F," but that's
            # presumably an accidental duplication?
            return "6"
        elif explicit_aa in "FW":
            # NOTE: Jurgen didn't specify a letter for this
            # category
            return "7"
        elif explicit_aa in "P":
            return "8"
        else:
            return "*"
    else:
        raise NotImplementedError(f"{method=} not supported")
