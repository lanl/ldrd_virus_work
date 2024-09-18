from viral_seq import run_workflow as workflow
from matplotlib.testing.decorators import image_comparison
import numpy as np
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from sklearn.utils.validation import check_is_fitted


@image_comparison(
    baseline_images=["test_optimization_plotting"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_optimization_plotting(tmp_path):
    rng = np.random.default_rng(seed=2024)
    data = {
        "Classifier1": rng.uniform(size=30),
        "Classifier2": rng.uniform(size=10),
        "Classifier3": rng.uniform(size=51),
    }
    workflow.optimization_plots(
        data,
        str(tmp_path / "test_optimization_plotting.csv"),
        str(tmp_path / "test_optimization_plotting.png"),
    )


@pytest.mark.parametrize("extract", [True, False])
def test_get_test_features(extract, tmpdir):
    raw_file = files("viral_seq.tests.inputs").joinpath(
        "get_test_features_test_file.csv"
    )
    test_file = str(raw_file)
    X_train = pd.read_csv(
        files("viral_seq.tests.inputs").joinpath("get_test_features_X_train.csv")
    )
    table_loc_test = str(
        files("viral_seq.tests.inputs") / "get_test_features_table_loc_test"
    )
    extract_cookie = raw_file if extract else files("viral_seq.tests") / "fake_file.dat"
    with tmpdir.as_cwd():
        with ExitStack() as stack:
            if extract:
                stack.enter_context(pytest.raises(NameError, match="table_info"))
            X_test, y_test = workflow.get_test_features(
                table_loc_test,
                "X_test.parquet.gzip",
                test_file,
                X_train,
                extract_cookie,
                debug=True,
            )
    if not extract:
        X_expected = pd.read_csv(
            files("viral_seq.tests.expected") / "get_test_features_X_expected.csv"
        )
        y_expected = pd.read_csv(
            files("viral_seq.tests.expected") / "get_test_features_y_expected.csv"
        )["Human Host"]
        assert_frame_equal(X_test, X_expected)
        assert_series_equal(y_test, y_expected)


def test_csv_conversion():
    input_csv = files("viral_seq.data") / "receptor_training.csv"
    postprocessed_df = workflow.csv_conversion(input_csv)
    assert_array_equal(
        postprocessed_df.columns,
        [
            "Species",
            "Accessions",
            "Human Host",
            "Is_Integrin",
            "Is_Sialic_Acid",
            "Is_Both",
        ],
    )
    assert postprocessed_df.shape == (94, 6)
    assert postprocessed_df.sum().Is_Integrin == 45
    assert postprocessed_df.sum().Is_Sialic_Acid == 53
    assert postprocessed_df.sum().Is_Both == 4


def test_train_rfc():
    data_header = [
        "kmer_AA_AACDEFG",
        "kmer_AA_ADTWKST",
        "kmer_PC_AELKTSR",
        "kmer_AA_AACDEFG",
        "kmer_AA_AACADAG",
        "kmer_PC_BACCEFG",
        "kmer_AA_BATGKGT",
        "kmer_PC_GALKTSR",
        "kmer_AA_GGCDEFG",
        "kmer_PC_GCCAGGE",
    ]
    data_values = np.zeros((10, 10))
    train_data = pd.DataFrame(data_values, columns=data_header)
    data_target = pd.Series(
        np.array((True, False, True, False, True, False, True, False, True, False)),
        name="Test",
    )

    file_folder = files("viral_seq.tests.expected")
    file_paths = [item for item in file_folder.iterdir()]
    out_path = [file_paths[-1].parent]

    clfr, topN, train, df = workflow.train_rfc(
        train_data,
        data_target,
        n_folds=5,
        paths=out_path,
        target_column="Test",
        random_state=0,
    )

    topN_df = pd.DataFrame(topN)

    topN_expected = pd.read_csv(
        files("viral_seq.tests.expected") / ("test_train_rfc_expected_topN.csv"),
        index_col=[0],
    )
    df_expected = pd.read_csv(
        files("viral_seq.tests.expected") / ("test_train_rfc_expected_df.csv"),
        index_col=[0],
    )

    topN_expected.columns = topN_expected.columns.astype(int)
    expected_train_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    assert_frame_equal(topN_df, topN_expected)
    assert_frame_equal(df, df_expected)
    assert_array_equal(train, expected_train_idx)
    check_is_fitted(clfr)
