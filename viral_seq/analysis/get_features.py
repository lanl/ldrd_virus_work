import re

# hard-coded lookup tables
codontab = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TGT": "C",
    "TGC": "C",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    "TAA": "STOP",
    "TGA": "STOP",
    "TAG": "STOP",
}
# Currently unused, but may be useful later
aminotab = {
    "F": ["TTT", "TTC"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "Y": ["TAT", "TAC"],
    "C": ["TGT", "TGC"],
    "W": ["TGG"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "H": ["CAT", "CAC"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "I": ["ATT", "ATC", "ATA"],
    "M": ["ATG"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "STOP": ["TAA", "TGA", "TAG"],
}


# Calculates all genomic features for a given record
def get_genomic_features(record):
    coding_dict = {}
    coding_pairs = []
    bridge_pairs = []
    nonbridge_pairs = []
    all_cnt_dict = {}
    coding_cnt_dict = {}
    bridge_cnt_dict = {}
    codons = []
    # get cds for each protein so we can count bias
    for feature in record.features:
        if feature.type == "CDS":
            # need to use extract because location can be complicated
            this_seq = feature.location.extract(record.seq)
            # add pairs to lists
            (
                these_coding_pairs,
                these_bridge_pairs,
                these_nonbridge_pairs,
                these_codons,
            ) = split_seq(str(this_seq), coding=True)
            coding_pairs = coding_pairs + these_coding_pairs
            bridge_pairs = bridge_pairs + these_bridge_pairs
            nonbridge_pairs = nonbridge_pairs + these_nonbridge_pairs
            codons = codons + these_codons
            # store information about these sequences need for calculating bias
            coding_cnt_dict = get_cnt_dict(str(this_seq), coding_dict)
            bridge_cnt_dict = get_cnt_dict("".join(these_bridge_pairs), bridge_cnt_dict)
    all_pairs = split_seq(str(record.seq), coding=False)[0]
    all_cnt_dict = get_cnt_dict(str(record.seq))
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
    return {**coding_ret, **all_ret, **bridge_ret, **nonbridge_ret, **codon_ret}


# Keeps a running count of the sequence length and number of A,T,G,C
def get_cnt_dict(seq: str, cnt_dict=None):
    if cnt_dict is None:
        cnt_dict = {}
    if "seq_len" in cnt_dict:
        cnt_dict["seq_len"] = cnt_dict["seq_len"] + len(seq)
    else:
        cnt_dict["seq_len"] = len(seq)
    if "A" in cnt_dict:
        cnt_dict["A"] = cnt_dict["A"] + float(len(re.findall(r"A", seq.upper())))
    else:
        cnt_dict["A"] = float(len(re.findall(r"A", seq.upper())))
    if "T" in cnt_dict:
        cnt_dict["T"] = cnt_dict["T"] + float(len(re.findall(r"T", seq.upper())))
    else:
        cnt_dict["T"] = float(len(re.findall(r"T", seq.upper())))
    if "G" in cnt_dict:
        cnt_dict["G"] = cnt_dict["G"] + float(len(re.findall(r"G", seq.upper())))
    else:
        cnt_dict["G"] = float(len(re.findall(r"G", seq.upper())))
    if "C" in cnt_dict:
        cnt_dict["C"] = cnt_dict["C"] + float(len(re.findall(r"C", seq.upper())))
    else:
        cnt_dict["C"] = float(len(re.findall(r"C", seq.upper())))
    return cnt_dict


# Pair list should include all pairs for this calculation
# Eg. if this is calculating dinucleotide bias for
# bridge pairs, it should include all bridge pairs for all
# coding regions for this record
# Calculation as defined in doi: 10.1126/science.aap9072
def get_dinucleotide_bias(pair_lst, cnt_dict, key_prefix=""):
    dinuc_dict = {}
    "".join(pair_lst)
    num_char = float(cnt_dict["seq_len"])
    num_pairs = float(len(pair_lst))
    if num_char == 0 or num_pairs == 0:
        return dinuc_dict
    # we can make the multiplicative factor to save time
    dinuc_factor = {}
    ATGC = ["A", "T", "G", "C"]
    for a in ATGC:
        for b in ATGC:
            dinuc_factor[a + b] = (num_char * num_char) / (
                cnt_dict[a] * cnt_dict[b] * num_pairs
            )
    # store bias value as calculated
    for pair in pair_lst:
        this_key = key_prefix + pair
        this_factor = dinuc_factor[pair] if pair in dinuc_factor else 0.0
        if this_key in dinuc_dict:
            dinuc_dict[this_key] = dinuc_dict[this_key] + this_factor
        else:
            dinuc_dict[this_key] = this_factor
    return dinuc_dict


# Calculation as defined in doi: 10.1126/science.aap9072
def get_codon_amino_bias(codons):
    codon_count = {}
    amino_count = {}
    total_amino = 0.0
    bias_ret = {}
    for codon in codons:
        if codon in codon_count:
            codon_count[codon] = codon_count[codon] + 1.0
        else:
            codon_count[codon] = 1.0
        amino = codontab[codon]
        total_amino = total_amino + 1.0  # this includes STOP codon as defined in ref
        if amino in amino_count:
            amino_count[amino] = amino_count[amino] + 1.0
        else:
            amino_count[amino] = 1.0
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
