from viral_seq.analysis.get_features import get_genomic_features
from viral_seq.analysis.spillover_predict import _append_recs
import pandas as pd
from pandas.testing import assert_frame_equal
from importlib.resources import files


def test_features():
    tests_dir = files("viral_seq") / "tests"
    test_record = _append_recs(tests_dir)
    # Calculate features for NC_019843.3 (MERS-CoV)
    df = pd.DataFrame(
        get_genomic_features([test_record]), index=["NC_019843.3"]
    ).reset_index()
    # Check our calculation matches published results in
    # https://doi.org/10.1371/journal.pbio.3001390
    df_expected = pd.read_csv(
        files("viral_seq.tests").joinpath("MERS-CoV_features.csv"), sep="\t"
    )
    assert_frame_equal(
        df.sort_index(axis=1),
        df_expected.sort_index(axis=1),
        check_names=True,
        rtol=1e-9,
        atol=1e-9,
    )
