from viral_seq.data import make_alternate_datasets as make_alts
import pandas as pd
from pandas.testing import assert_frame_equal
from importlib.resources import files
import pytest
import numpy as np
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


# Utilizing some pre-existing accessions in the test folder
@pytest.mark.parametrize(
    "cache, accession, train_accessions, expected_result",
    [
        (
            "cache",
            "NC_033698.1",
            None,
            {"partial": [], "duplicate": [], "no_good_cds": []},
        ),
        (
            "cache",
            "HM119401.1",
            {"NC_033698.1"},
            {"partial": [], "duplicate": [], "no_good_cds": []},
        ),
        (
            "cache",
            "NC_033698.1",
            {"NC_033698.1"},
            {"partial": [], "duplicate": [0], "no_good_cds": []},
        ),
        (
            "cache_unfiltered",
            "HM147992.1",
            None,
            {"partial": [0], "duplicate": [], "no_good_cds": []},
        ),
        (
            "caches/no_good_cds",
            "KY312541.1",
            None,
            {"partial": [0], "duplicate": [], "no_good_cds": [0]},
        ),
    ],
)
def test_get_bad_indexes(cache, accession, train_accessions, expected_result):
    this_cache = files("viral_seq.tests") / cache
    df = pd.DataFrame(
        [["0", accession, "Virus"]], columns=["Unnamed: 0", "Accessions", "Species"]
    )
    result = make_alts.get_bad_indexes(df, this_cache, train_accessions)
    for key in result:
        assert result[key] == expected_result[key]


def test_shuffle_regression():
    exp_order = [19, 9, 5, 6, 18, 13, 12, 2, 3, 4, 8, 7, 16, 15, 17, 0, 14, 11, 10, 1]
    train_data = files("viral_seq.tests") / "TrainingSet.csv"
    test_data = files("viral_seq.tests") / "TestSet.csv"
    df_train = pd.read_csv(train_data, index_col=0)
    df_test = pd.read_csv(test_data, index_col=0)
    df_train_shuffled, df_test_shuffled = make_alts.shuffle(
        df_train, df_test, random_state=123456
    )
    actual_data = pd.concat([df_train_shuffled, df_test_shuffled]).reset_index(
        drop=True
    )
    expected_data = (
        pd.concat([df_train, df_test]).iloc[exp_order].reset_index(drop=True)
    )
    assert_frame_equal(actual_data, expected_data)


@given(
    random_state=hnp.from_dtype(np.dtype(np.uint32)), different_balance=st.booleans()
)
def test_shuffle_property(random_state, different_balance):
    """Test the properties that should be perserved by the shuffle."""
    # train/test sets are half human host, 10 samples
    train_data = files("viral_seq.tests") / "TrainingSet.csv"
    test_data = files("viral_seq.tests") / "TestSet.csv"
    df_train = pd.read_csv(train_data, index_col=0)
    df_test = pd.read_csv(test_data, index_col=0)
    if different_balance:
        # make train human host ratio and number of samples not match test
        df_train = df_train.loc[df_train["Human Host"]]
    df_train_shuffled, df_test_shuffled = make_alts.shuffle(
        df_train, df_test, random_state
    )

    # assert counts of "Human Host" values haven't changed for each set
    for df1, df2 in zip([df_train_shuffled, df_test_shuffled], [df_train, df_test]):
        assert df1["Human Host"].sum() == df2["Human Host"].sum()
        assert df1.shape[0] == df2.shape[0]

    # assert we still have the same data present
    original_data = pd.concat([df_train, df_test]).sort_values(
        by="Species", ignore_index=True
    )
    shuffled_data = pd.concat([df_train_shuffled, df_test_shuffled]).sort_values(
        by="Species", ignore_index=True
    )
    assert_frame_equal(original_data, shuffled_data, check_dtype=False)
