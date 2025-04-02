import uuid
from viral_seq.data import make_alternate_datasets as make_alts
import viral_seq.data.make_data_summary_plots as data_summary
import pandas as pd
from pandas.testing import assert_frame_equal
from importlib.resources import files
import pytest
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from matplotlib.testing.compare import compare_images


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
    """Test the properties that should be preserved by the shuffle."""
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
    assert_frame_equal(original_data, shuffled_data)


@pytest.mark.parametrize(
    "df, families_exp",
    [
        (
            pd.read_csv(str(files("viral_seq.tests") / "TrainingSet.csv"))[["Species"]],
            {
                "Flaviviridae": 3,
                "Togaviridae": 2,
                "Phenuiviridae": 1,
                "Papillomaviridae": 3,
                "Peribunyaviridae": 1,
            },
        ),
        (
            pd.DataFrame({"Species": ["Goose coronavirus CB17"]}),
            {"Coronaviridae": 1},
        ),  # this is in corrections
    ],
)
def test_get_family_counts(df, families_exp):
    families = data_summary._get_family_counts(df)
    assert families == families_exp


def test_get_family_counts_missing():
    df = pd.DataFrame({"Species": ["Missing Virus"]})
    with pytest.raises(
        ValueError, match="Couldn't find taxonomy for the following viruses:"
    ):
        data_summary._get_family_counts(df)


@pytest.mark.parametrize("target_column", ["Human Host", "other"])
def test_plot_family_heatmap(tmpdir, target_column):
    expected_plot = files("viral_seq.tests.expected") / (
        f"test_plot_family_heatmap_{target_column}.png".replace(" ", "_")
    )
    expected_data = pd.read_csv(
        files("viral_seq.tests.expected")
        / (f"test_plot_family_heatmap_{target_column}.csv".replace(" ", "_")),
        index_col=0,
    )
    rng = np.random.default_rng(12345)
    train_file = str(files("viral_seq.tests") / "TrainingSet.csv")
    test_file = str(files("viral_seq.tests") / "TestSet.csv")
    df_train = pd.read_csv(train_file, index_col=0)
    df_test = pd.read_csv(test_file, index_col=0)
    df_train[target_column] = rng.integers(0, 2, size=len(df_train), dtype=bool)
    df_test[target_column] = rng.integers(0, 2, size=len(df_test), dtype=bool)
    with tmpdir.as_cwd():
        df_train.to_csv("train_file.csv")
        df_test.to_csv("test_file.csv")
        data_summary.plot_family_heatmap(
            "train_file.csv", "test_file.csv", target_column
        )
        assert_frame_equal(
            pd.read_csv("plot_family_heatmap.csv", index_col=0), expected_data
        )
        assert compare_images(expected_plot, "plot_family_heatmap.png", 0.001) is None


def test__plot_family_heatmap(tmpdir):
    family_counts = pd.DataFrame(
        {
            "Phenuiviridae": {
                "Train Human Host": 1,
                "Train Not Human Host": 0,
                "Test Human Host": 2,
                "Test Not Human Host": 3,
            },
            "Flaviviridae": {
                "Train Human Host": 2,
                "Train Not Human Host": 1,
                "Test Human Host": 0,
                "Test Not Human Host": 0,
            },
        }
    )
    expected_plot = files("viral_seq.tests.expected") / "test__plot_family_heatmap.png"
    expected_data = pd.read_csv(
        files("viral_seq.tests.expected") / "test__plot_family_heatmap.csv", index_col=0
    )
    with tmpdir.as_cwd():
        data_summary._plot_family_heatmap(family_counts)
        assert_frame_equal(
            pd.read_csv("plot_family_heatmap.csv", index_col=0), expected_data
        )
        assert compare_images(expected_plot, "plot_family_heatmap.png", 0.001) is None


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            # when one viral family is completely
            # absent from one of the splits, the
            # KL divergence is infinite
            {
                "Train Human Host": [1, 2],
                "Train Not Human Host": [0, 1],
                "Test Human Host": [2, 0],
                "Test Not Human Host": [3, 0],
            },
            np.inf,
        ),
        (
            # when the viral families are perfectly
            # balanced across the splits, the KL
            # divergence is zero
            {
                "Train Human Host": [2, 1],
                "Train Not Human Host": [3, 1],
                "Test Human Host": [2, 1],
                "Test Not Human Host": [3, 1],
            },
            0.0,
        ),
    ],
)
def test_relative_entropy(tmpdir, data, expected):
    df = pd.DataFrame.from_dict(
        data, columns=["viral family 1", "viral family 2"], orient="index"
    )
    with tmpdir.as_cwd():
        fname = f"{uuid.uuid4()}.csv"
        df.to_csv(fname)
        actual = data_summary.relative_entropy_viral_families(heatmap_csv=fname)
    assert_allclose(actual, expected)


def test_relabel_dataset():
    """test the output of relabeling the dataset from 'Human Host' to 'human', 'primate', and 'mammal'"""
    # input datasets include relabeled data, train, and test files
    train_data = files("viral_seq.tests") / "TrainingSet.csv"
    test_data = files("viral_seq.tests") / "TestSet.csv"
    relabeled_data = files("viral_seq.tests.expected") / "relabeled_data_exp.npz"
    df_train = pd.read_csv(train_data, index_col=0)
    df_test = pd.read_csv(test_data, index_col=0)
    df_relabeled = np.load(relabeled_data)

    relabeled_train_exp = (
        files("viral_seq.tests.expected") / "relabeled_df_train_exp.csv"
    )
    relabeled_test_exp = files("viral_seq.tests.expected") / "relabeled_df_test_exp.csv"
    relabeled_train_df_exp = pd.read_csv(relabeled_train_exp)
    relabeled_test_df_exp = pd.read_csv(relabeled_test_exp)

    # relabel input datasets
    relabeled_df_train, relabeled_df_test = make_alts.make_relabeled_dataset(
        df_relabeled, df_train, df_test
    )

    # sanity check to see if virus that flipped during relabeling labeled correctly in output
    # Akhmeta virus: human -> mammal
    assert (
        relabeled_df_test[relabeled_df_test["Species"] == "Akhmeta virus"][
            "human"
        ].item()
        is False
    )
    assert_frame_equal(relabeled_df_train, relabeled_train_df_exp)
    assert_frame_equal(relabeled_df_test, relabeled_test_df_exp)
