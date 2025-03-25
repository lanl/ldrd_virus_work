from viral_seq.analysis import biological_analysis as ba
import pytest
from pandas.testing import assert_frame_equal
import pandas as pd


@pytest.mark.parametrize(
    "test_kmers, mapping_method, exp_out",
    [
        (
            ["kmer_AA_AGC"],
            "jurgen_schmidt",
            {0: {"kmer_AA_AGC": ("AGC", 1.3)}},
        ),
        (
            ["kmer_PC_057"],
            "jurgen_schmidt",
            {
                0: {"kmer_PC_057": ("AKF", 0.2333333333333334)},
                1: {"kmer_PC_057": ("AKW", -0.9999999999999999)},
                2: {"kmer_PC_057": ("ARF", 0.033333333333333215)},
                3: {"kmer_PC_057": ("ARW", -1.2)},
                4: {"kmer_PC_057": ("GKF", -0.5)},
                5: {"kmer_PC_057": ("GKW", -1.7333333333333334)},
                6: {"kmer_PC_057": ("GRF", -0.7000000000000002)},
                7: {"kmer_PC_057": ("GRW", -1.9333333333333336)},
            },
        ),
        (
            ["kmer_PC_ABC"],
            "shen_2007",
            {
                0: {"kmer_PC_ABC": ("ACF", 2.3666666666666667)},
                1: {"kmer_PC_ABC": ("ACL", 2.6999999999999997)},
                2: {"kmer_PC_ABC": ("ACI", 2.9333333333333336)},
                3: {"kmer_PC_ABC": ("ACP", 0.8999999999999999)},
                4: {"kmer_PC_ABC": ("GCF", 1.6333333333333335)},
                5: {"kmer_PC_ABC": ("GCL", 1.9666666666666668)},
                6: {"kmer_PC_ABC": ("GCI", 2.1999999999999997)},
                7: {"kmer_PC_ABC": ("GCP", 0.16666666666666666)},
                8: {"kmer_PC_ABC": ("VCF", 3.1666666666666665)},
                9: {"kmer_PC_ABC": ("VCL", 3.5)},
                10: {"kmer_PC_ABC": ("VCI", 3.733333333333333)},
                11: {"kmer_PC_ABC": ("VCP", 1.7)},
            },
        ),
    ],
)
def test_hydrophobicity_score(test_kmers, mapping_method, exp_out):
    hydro_score = ba.hydrophobicity_score(test_kmers, mapping_method)
    exp_out_df = pd.DataFrame.from_dict(exp_out)
    assert_frame_equal(hydro_score, exp_out_df)


@pytest.mark.parametrize(
    "peptide, mapping_method, exp_out",
    [
        (
            "012",
            "jurgen_schmidt",
            ["ACS", "ACT", "ACY", "GCS", "GCT", "GCY"],
        ),
        (
            "ABC",
            "shen_2007",
            [
                "ACF",
                "ACL",
                "ACI",
                "ACP",
                "GCF",
                "GCL",
                "GCI",
                "GCP",
                "VCF",
                "VCL",
                "VCI",
                "VCP",
            ],
        ),
    ],
)
def test_pc_to_aa(peptide, mapping_method, exp_out):
    kmers_out = ba.pc_to_aa(peptide, mapping_method)
    assert kmers_out == exp_out
