from viral_seq import run_workflow as workflow
from viral_seq.analysis import classifier
import numpy as np
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib.testing.compare import compare_images
import os
import json


@pytest.mark.parametrize(
    "group, random_state, get_hyperparameters_return, cv_score_return, expected_results",
    [
        # test each classifier type gets the appropriate default params
        (
            "RandomForestClassifier",
            123,
            {"targets": [0.5], "target": 0.5, "params": {}},
            0.8,
            {
                "targets": [0.8, 0.5],
                "target": 0.8,
                "params": {"n_estimators": 2000, "n_jobs": 1, "random_state": 123},
            },
        ),
        (
            "ExtraTreesClassifier",
            123,
            {"targets": [0.5], "target": 0.5, "params": {}},
            0.8,
            {
                "targets": [0.8, 0.5],
                "target": 0.8,
                "params": {
                    "bootstrap": True,
                    "n_estimators": 2000,
                    "n_jobs": 1,
                    "random_state": 123,
                },
            },
        ),
        (
            "LGBMClassifier Dart",
            123,
            {"targets": [0.5], "target": 0.5, "params": {}},
            0.8,
            {
                "targets": [0.8, 0.5],
                "target": 0.8,
                "params": {
                    "verbose": -1,
                    "force_col_wise": True,
                    "n_estimators": 500,
                    "n_jobs": 1,
                    "boosting_type": "dart",
                    "random_state": 123,
                },
            },
        ),
        (
            "LGBMClassifier Boost",
            123,
            {"targets": [0.5], "target": 0.5, "params": {}},
            0.8,
            {
                "targets": [0.8, 0.5],
                "target": 0.8,
                "params": {
                    "verbose": -1,
                    "force_col_wise": True,
                    "n_estimators": 500,
                    "n_jobs": 1,
                    "random_state": 123,
                },
            },
        ),
        (
            "XGBClassifier Boost",
            123,
            {"targets": [0.5], "target": 0.5, "params": {}},
            0.8,
            {
                "targets": [0.8, 0.5],
                "target": 0.8,
                "params": {"n_jobs": 1, "random_state": 123},
            },
        ),
        # optimization is better case; we are just returned the params from optimization
        (
            "RandomForestClassifier",
            123,
            {
                "targets": [0.8],
                "target": 0.8,
                "params": {
                    "n_estimators": 2000,
                    "n_jobs": 1,
                    "random_state": 123,
                    "max_features": 0.5,
                },
            },
            0.5,
            {
                "targets": [0.5, 0.8],
                "target": 0.8,
                "params": {
                    "n_estimators": 2000,
                    "n_jobs": 1,
                    "random_state": 123,
                    "max_features": 0.5,
                },
            },
        ),
    ],
)
def test_optimize_model(
    tmpdir,
    mocker,
    group,
    random_state,
    get_hyperparameters_return,
    cv_score_return,
    expected_results,
):
    name = f"{group} Seed:{random_state}"
    optimize_args = classifier.get_model_arguments(1, random_state, 1, 1)[name][
        "optimize"
    ]
    # these will not be used
    model = mocker.MagicMock()
    X_train = mocker.MagicMock()
    y_train = mocker.MagicMock()
    # these functions would be slow to run and are separately tested
    mocker.patch(
        "viral_seq.analysis.classifier.get_hyperparameters",
        return_value=get_hyperparameters_return,
    )
    mocker.patch("viral_seq.analysis.classifier.cv_score", return_value=cv_score_return)
    with tmpdir.as_cwd():
        results = workflow.optimize_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            outfile="outfile.json",
            config=optimize_args["config"],
            num_samples=optimize_args["num_samples"],
            optimize="yes",
            name=name,
            debug=True,
            random_state=random_state,
            n_jobs_cv=optimize_args["n_jobs_cv"],
            n_jobs=1,
        )
        assert os.path.exists("outfile.json")
    assert results == expected_results


@pytest.mark.parametrize(
    "group, random_state, get_hyperparameters_return, cv_score_return",
    [
        # test each classifier type gets the appropriate default params
        (
            "RandomForestClassifier",
            123,
            {"targets": [0.5], "target": 0.5, "params": {}},
            0.5,
        )
    ],
)
def test_optimize_model_debug_fail(
    tmpdir,
    mocker,
    group,
    random_state,
    get_hyperparameters_return,
    cv_score_return,
):
    name = f"{group} Seed:{random_state}"
    optimize_args = classifier.get_model_arguments(1, random_state, 1, 1)[name][
        "optimize"
    ]
    # these will not be used
    model = mocker.MagicMock()
    X_train = mocker.MagicMock()
    y_train = mocker.MagicMock()
    # these functions would be slow to run and are separately tested
    mocker.patch(
        "viral_seq.analysis.classifier.get_hyperparameters",
        return_value=get_hyperparameters_return,
    )
    mocker.patch("viral_seq.analysis.classifier.cv_score", return_value=cv_score_return)
    with tmpdir.as_cwd(), pytest.raises(
        AssertionError, match="ROC AUC achieved is too poor"
    ):
        workflow.optimize_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            outfile="outfile.json",
            config=optimize_args["config"],
            num_samples=optimize_args["num_samples"],
            optimize="yes",
            name=name,
            debug=True,
            random_state=random_state,
            n_jobs_cv=optimize_args["n_jobs_cv"],
            n_jobs=1,
        )


def test_optimize_model_skip(tmpdir, mocker):
    params = {
        "targets": [0.8, 0.5],
        "target": 0.8,
        "params": {"n_estimators": 2000, "n_jobs": 1, "random_state": 123},
    }
    with tmpdir.as_cwd():
        with open("outfile.json", "w") as f:
            json.dump(params, f)
        # for skip the function should just load the outfile;
        # most of the other parameters aren't used
        results = workflow.optimize_model(
            model=mocker.MagicMock(),
            X_train=mocker.MagicMock(),
            y_train=mocker.MagicMock(),
            outfile="outfile.json",
            config=mocker.MagicMock(),
            num_samples=0,
            optimize="skip",
            name="",
            debug=True,
            random_state=0,
            n_jobs_cv=1,
            n_jobs=1,
        )
    assert results == params


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


def test_plot_confusion_matrices(tmpdir):
    # minimum number of images to test for full coverage
    exp_model0_plot = files("viral_seq.tests.expected").joinpath(
        "test_Model_0_confusion_matrix.png"
    )
    exp_model1_plot = files("viral_seq.tests.expected").joinpath(
        "test_Model_1_confusion_matrix.png"
    )
    exp_group_plot = files("viral_seq.tests.expected").joinpath(
        "test_Group_confusion_matrix.png"
    )
    exp_ensemble_plot = files("viral_seq.tests.expected").joinpath(
        "test_Ensemble_confusion_matrix.png"
    )
    rng = np.random.default_rng(123)
    y_test = rng.integers(2, size=10)
    model_arguments = {f"Model {i}": {"group": "Group"} for i in range(2)}
    predictions_ensemble_hard_eer = {
        name: rng.integers(2, size=10) for name in model_arguments
    }
    comp_names_ensembles = ["Ensemble"]
    comp_preds_ensembles = [rng.integers(2, size=10)]
    workflow._plot_confusion_matrices(
        y_test,
        model_arguments,
        predictions_ensemble_hard_eer,
        comp_names_ensembles,
        comp_preds_ensembles,
        tmpdir,
    )
    with tmpdir.as_cwd():
        assert (
            compare_images(exp_model0_plot, "Model_0_confusion_matrix.png", 0.001)
            is None
        )
        assert (
            compare_images(exp_model1_plot, "Model_1_confusion_matrix.png", 0.001)
            is None
        )
        assert (
            compare_images(exp_group_plot, "Group_confusion_matrix.png", 0.001) is None
        )
        assert (
            compare_images(exp_ensemble_plot, "Ensemble_confusion_matrix.png", 0.001)
            is None
        )


def test_plot_confusion_matrices_strict(tmpdir):
    """Ensure errors if len(comp_names_ensembles) != len(comp_preds_ensembles)."""
    y_test = np.ones(10)
    model_arguments = {f"Model {i}": {"group": "Group"} for i in range(2)}
    predictions_ensemble_hard_eer = {name: np.zeros(10) for name in model_arguments}
    comp_names_ensembles = [f"Ensemble {i}" for i in range(2)]
    comp_preds_ensembles = [np.zeros(10) for i in range(3)]
    with pytest.raises(
        ValueError, match=r"zip\(\) argument 2 is longer than argument 1"
    ):
        workflow._plot_confusion_matrices(
            y_test,
            model_arguments,
            predictions_ensemble_hard_eer,
            comp_names_ensembles,
            comp_preds_ensembles,
            tmpdir,
        )
