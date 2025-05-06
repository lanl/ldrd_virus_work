from viral_seq.analysis import biological_analysis as ba
import pytest
from pandas.testing import assert_frame_equal
import pandas as pd


@pytest.mark.parametrize(
    "test_kmers, mapping_method, exp_out",
    [  # test case for single AA-kmer
        (
            ["kmer_AA_AGC"],
            None,
            {0: {"kmer_AA_AGC": ("AGC", 1.3)}},
        ),
        # test case for PC-kmer with "jurgen_schmidt" mapping method
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
        # test case for PC-kmer with "shen_2007" mapping method
        (
            ["kmer_PC_123"],
            "shen_2007",
            {
                0: {"kmer_PC_123": ("ACF", 2.3666666666666667)},
                1: {"kmer_PC_123": ("ACL", 2.6999999999999997)},
                2: {"kmer_PC_123": ("ACI", 2.9333333333333336)},
                3: {"kmer_PC_123": ("ACP", 0.8999999999999999)},
                4: {"kmer_PC_123": ("GCF", 1.6333333333333335)},
                5: {"kmer_PC_123": ("GCL", 1.9666666666666668)},
                6: {"kmer_PC_123": ("GCI", 2.1999999999999997)},
                7: {"kmer_PC_123": ("GCP", 0.16666666666666666)},
                8: {"kmer_PC_123": ("VCF", 3.1666666666666665)},
                9: {"kmer_PC_123": ("VCL", 3.5)},
                10: {"kmer_PC_123": ("VCI", 3.733333333333333)},
                11: {"kmer_PC_123": ("VCP", 1.7)},
            },
        ),
        # test case for mix of PC and AA kmers
        (
            ["kmer_AA_AGC", "kmer_PC_012"],
            "jurgen_schmidt",
            {
                0: {
                    "kmer_PC_012": ("ACS", 1.1666666666666667),
                    "kmer_AA_AGC": ("AGC", 1.3),
                },
                1: {"kmer_PC_012": ("ACT", 1.2), "kmer_AA_AGC": None},
                2: {"kmer_PC_012": ("ACY", 1.0), "kmer_AA_AGC": None},
                3: {"kmer_PC_012": ("GCS", 0.43333333333333335), "kmer_AA_AGC": None},
                4: {"kmer_PC_012": ("GCT", 0.46666666666666673), "kmer_AA_AGC": None},
                5: {"kmer_PC_012": ("GCY", 0.26666666666666666), "kmer_AA_AGC": None},
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
            "123",
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


@pytest.mark.parametrize(
    "kmer_in, exp_gravy",
    # these example AA peptide sequences taken from
    # https://doi.org/10.3390/ijms25042431
    [
        (
            ["kmer_AA_TKVIPYVRYL"],
            0.34,
        ),
        (
            ["kmer_AA_IQPKTKVIPYVRYL"],
            -0.08,
        ),
        (
            ["kmer_AA_LYQEPVLGPVRGPFPIIV"],
            0.67,
        ),
    ],
)
def test_gravy_calculation(kmer_in, exp_gravy):
    # sanity check for gravy hydrophobicity using known peptide-scores from literature
    gravy_score = ba.hydrophobicity_score(kmer_in)
    assert round(gravy_score.iloc[0][0][1], 2) == exp_gravy


def test_hydrophobicity_score_error():
    # test that the function enforces the requirement to provide a mapping method for translating PC kmers
    with pytest.raises(ValueError, match="Please provide a mapping method"):
        ba.hydrophobicity_score(["kmer_PC_123"])
