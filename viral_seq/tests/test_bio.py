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
            {"kmer": ["kmer_AA_AGC"], "peptide": ["AGC"], "score": [1.3]},
        ),
        # test case for PC-kmer with "jurgen_schmidt" mapping method
        (
            ["kmer_PC_057"],
            "jurgen_schmidt",
            {
                "kmer": ["kmer_PC_057"] * 8,
                "peptide": [
                    "GRW",
                    "GRF",
                    "GKW",
                    "GKF",
                    "ARW",
                    "ARF",
                    "AKW",
                    "AKF",
                ],
                "score": [
                    -1.9333333333333336,
                    -0.7000000000000002,
                    -1.7333333333333334,
                    -0.5,
                    -1.2,
                    0.033333333333333215,
                    -0.9999999999999999,
                    0.2333333333333334,
                ],
            },
        ),
        # test case for PC-kmer with "shen_2007" mapping method
        (
            ["kmer_PC_123"],
            "shen_2007",
            {
                "kmer": ["kmer_PC_123"] * 12,
                "peptide": [
                    "VCP",
                    "VCI",
                    "VCL",
                    "VCF",
                    "GCP",
                    "GCI",
                    "GCL",
                    "GCF",
                    "ACP",
                    "ACI",
                    "ACL",
                    "ACF",
                ],
                "score": [
                    1.7,
                    3.733333333333333,
                    3.5,
                    3.1666666666666665,
                    0.16666666666666666,
                    2.1999999999999997,
                    1.9666666666666668,
                    1.6333333333333335,
                    0.8999999999999999,
                    2.9333333333333336,
                    2.6999999999999997,
                    2.3666666666666667,
                ],
            },
        ),
        # test case for mix of PC and AA kmers
        (
            ["kmer_AA_AGC", "kmer_PC_012"],
            "jurgen_schmidt",
            {
                "kmer": ["kmer_PC_012"] * 6 + ["kmer_AA_AGC"],
                "peptide": [
                    "GCY",
                    "GCT",
                    "GCS",
                    "ACY",
                    "ACT",
                    "ACS",
                    "AGC",
                ],
                "score": [
                    0.26666666666666666,
                    0.46666666666666673,
                    0.43333333333333335,
                    1.0,
                    1.2,
                    1.1666666666666667,
                    1.3,
                ],
            },
        ),
    ],
)
def test_hydrophobicity_score(test_kmers, mapping_method, exp_out):
    hydro_score = ba.hydrophobicity_score(test_kmers, mapping_method)
    exp_out_df = pd.DataFrame.from_records(exp_out)
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
