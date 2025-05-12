from viral_seq.analysis import biological_analysis as ba
import pytest
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
import pandas as pd


@pytest.mark.parametrize(
    "test_kmers, mapping_method, exp_out",
    [  # test case for single AA-kmer
        (
            ["kmer_AA_AGC"],
            None,
            {"kmer": {0: "kmer_AA_AGC"}, "peptide": {0: "AGC"}, "score": {0: 1.3}},
        ),
        # test case for PC-kmer with "jurgen_schmidt" mapping method
        (
            ["kmer_PC_057"],
            "jurgen_schmidt",
            {
                "kmer": {
                    7: "kmer_PC_057",
                    6: "kmer_PC_057",
                    5: "kmer_PC_057",
                    4: "kmer_PC_057",
                    3: "kmer_PC_057",
                    2: "kmer_PC_057",
                    1: "kmer_PC_057",
                    0: "kmer_PC_057",
                },
                "peptide": {
                    7: "GRW",
                    6: "GRF",
                    5: "GKW",
                    4: "GKF",
                    3: "ARW",
                    2: "ARF",
                    1: "AKW",
                    0: "AKF",
                },
                "score": {
                    7: -1.9333333333333336,
                    6: -0.7000000000000002,
                    5: -1.7333333333333334,
                    4: -0.5,
                    3: -1.2,
                    2: 0.033333333333333215,
                    1: -0.9999999999999999,
                    0: 0.2333333333333334,
                },
            },
        ),
        # test case for PC-kmer with "shen_2007" mapping method
        (
            ["kmer_PC_123"],
            "shen_2007",
            {
                "kmer": {
                    11: "kmer_PC_123",
                    10: "kmer_PC_123",
                    9: "kmer_PC_123",
                    8: "kmer_PC_123",
                    7: "kmer_PC_123",
                    6: "kmer_PC_123",
                    5: "kmer_PC_123",
                    4: "kmer_PC_123",
                    3: "kmer_PC_123",
                    2: "kmer_PC_123",
                    1: "kmer_PC_123",
                    0: "kmer_PC_123",
                },
                "peptide": {
                    11: "VCP",
                    10: "VCI",
                    9: "VCL",
                    8: "VCF",
                    7: "GCP",
                    6: "GCI",
                    5: "GCL",
                    4: "GCF",
                    3: "ACP",
                    2: "ACI",
                    1: "ACL",
                    0: "ACF",
                },
                "score": {
                    11: 1.7,
                    10: 3.733333333333333,
                    9: 3.5,
                    8: 3.1666666666666665,
                    7: 0.16666666666666666,
                    6: 2.1999999999999997,
                    5: 1.9666666666666668,
                    4: 1.6333333333333335,
                    3: 0.8999999999999999,
                    2: 2.9333333333333336,
                    1: 2.6999999999999997,
                    0: 2.3666666666666667,
                },
            },
        ),
        # test case for mix of PC and AA kmers
        (
            ["kmer_AA_AGC", "kmer_PC_012"],
            "jurgen_schmidt",
            {
                "kmer": {
                    6: "kmer_PC_012",
                    5: "kmer_PC_012",
                    4: "kmer_PC_012",
                    3: "kmer_PC_012",
                    2: "kmer_PC_012",
                    1: "kmer_PC_012",
                    0: "kmer_AA_AGC",
                },
                "peptide": {
                    6: "GCY",
                    5: "GCT",
                    4: "GCS",
                    3: "ACY",
                    2: "ACT",
                    1: "ACS",
                    0: "AGC",
                },
                "score": {
                    6: 0.26666666666666666,
                    5: 0.46666666666666673,
                    4: 0.43333333333333335,
                    3: 1.0,
                    2: 1.2,
                    1: 1.1666666666666667,
                    0: 1.3,
                },
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
    assert_allclose(gravy_score.score, exp_gravy, rtol=0.02)


def test_hydrophobicity_score_error():
    # test that the function enforces the requirement to provide a mapping method for translating PC kmers
    with pytest.raises(ValueError, match="Please provide a mapping method"):
        ba.hydrophobicity_score(["kmer_PC_123"])
