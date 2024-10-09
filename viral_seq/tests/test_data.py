from viral_seq.data import make_alternate_datasets as make_alts
import pandas as pd
from pandas.testing import assert_frame_equal
from importlib.resources import files
import pytest


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


@pytest.mark.parametrize("random_state", [123456, 203254, 987654])
def test_shuffle(random_state):
    train_data = files("viral_seq.tests") / "TrainingSet.csv"
    test_data = files("viral_seq.tests") / "TestSet.csv"
    df_train = pd.read_csv(train_data, index_col=0)
    df_test = pd.read_csv(test_data, index_col=0)
    df_train_shuffled, df_test_shuffled = make_alts.shuffle(
        df_train, df_test, random_state
    )

    # assert count of "Human Host" hasn't changed
    for df1, df2 in zip([df_train_shuffled, df_test_shuffled], [df_train, df_test]):
        for truth in [True, False]:
            assert (
                df1["Human Host"].value_counts()[truth]
                == df2["Human Host"].value_counts()[truth]
            )

    # assert we have changed the tables in some way
    with pytest.raises(AssertionError):
        assert_frame_equal(df_train, df_train_shuffled, check_dtype=False)
    with pytest.raises(AssertionError):
        assert_frame_equal(df_test, df_test_shuffled, check_dtype=False)

    # assert we still have the same data present
    original_data = pd.concat([df_train, df_test]).sort_values(
        by="Species", ignore_index=True
    )
    shuffled_data = pd.concat([df_train_shuffled, df_test_shuffled]).sort_values(
        by="Species", ignore_index=True
    )
    assert_frame_equal(original_data, shuffled_data, check_dtype=False)
