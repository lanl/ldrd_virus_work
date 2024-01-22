from viral_seq.analysis.get_features import get_genomic_features, get_kmers
from viral_seq.analysis.spillover_predict import _append_recs
import pandas as pd
from pandas.testing import assert_frame_equal
from importlib.resources import files
import pytest


@pytest.mark.parametrize(
    "accession, sep, file_name, calc_feats",
    [
        (
            "NC_019843.3",
            "\t",
            "MERS-CoV_features.csv",
            get_genomic_features,
        ),  # generic viral genomic feature test
        (
            "NC_007620.1",
            ",",
            "Menangle_features.csv",
            get_genomic_features,
        ),  # bad coding sequence test
        (
            "NC_007620.1",
            ",",
            "Menangle_features_kmers.csv",
            lambda e: get_kmers(e, k=2),
        ),  # bad coding sequence kmer calculation test
        (
            "HM045787.1",
            ",",
            "Chikungunya_features.csv",
            get_genomic_features,
        ),  # ambiguous nucleotide test
    ],
)
@pytest.mark.filterwarnings("error:Partial codon")
def test_features(accession, sep, file_name, calc_feats):
    tests_dir = files("viral_seq") / "tests" / accession
    test_record = _append_recs(tests_dir)
    df = pd.DataFrame(calc_feats([test_record]), index=[accession]).reset_index()
    # For viral genomic features check our calculation matches published results in
    # https://doi.org/10.1371/journal.pbio.3001390
    # For k-mers regression test
    df_expected = pd.read_csv(files("viral_seq.tests").joinpath(file_name), sep=sep)
    assert_frame_equal(
        df.sort_index(axis=1),
        df_expected.sort_index(axis=1),
        check_names=True,
        rtol=1e-9,
        atol=1e-9,
    )
