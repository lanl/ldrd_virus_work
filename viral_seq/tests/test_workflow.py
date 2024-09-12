from viral_seq import run_workflow as workflow
import numpy as np
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib.testing.compare import compare_images
from numpy.testing import assert_array_equal


def test_optimization_plotting(tmpdir):
    rng = np.random.default_rng(seed=2024)
    data = {
        "Classifier1": rng.uniform(size=30),
        "Classifier2": rng.uniform(size=10),
        "Classifier3": rng.uniform(size=51),
    }
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_optimization_plotting.png"
    )
    with tmpdir.as_cwd():
        workflow.optimization_plots(
            data,
            "test",
            tmpdir,
        )
        assert (
            compare_images(expected_plot, "test_optimization_plot.png", 0.001) is None
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
