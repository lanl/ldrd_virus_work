from viral_seq import run_workflow as workflow
import numpy as np
from numpy.testing import assert_allclose
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib.testing.compare import compare_images
from sklearn.ensemble import RandomForestClassifier, StackingClassifier


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


def test_plot_logistic_stacked_weights(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_logistic_stacked_weights.png"
    )
    rng = np.random.default_rng(2025)
    X = rng.random(size=(10, 10))
    y = rng.choice(2, 10)
    models = [
        (f"RF {i}", RandomForestClassifier(random_state=2025).fit(X, y))
        for i in range(2)
    ]
    stacked_logistic = StackingClassifier(models, cv="prefit").fit(X, y)
    estimator_names = [f"Name {i}" for i in range(2)]
    file_name = "Test.png"
    with tmpdir.as_cwd():
        workflow._plot_logistic_stacked_weights(
            stacked_logistic, estimator_names, file_name
        )
        assert compare_images(expected_plot, "Test.png", 0.001) is None


@pytest.mark.parametrize(
    "cv, exp_fpr, exp_tpr, exp_pred, exp_plot",
    [
        (
            5,
            np.array([0.0, 0.0, 0.0, 0.0, 0.33333333, 0.33333333, 1.0, 1.0]),
            np.array(
                [
                    0.0,
                    0.14285714,
                    0.42857143,
                    0.71428571,
                    0.71428571,
                    0.85714286,
                    0.85714286,
                    1.0,
                ]
            ),
            np.array(
                [
                    0.5023331,
                    0.50264211,
                    0.50094257,
                    0.49955203,
                    0.50109708,
                    0.50125158,
                    0.498625,
                    0.49800699,
                    0.5023331,
                    0.49924302,
                ]
            ),
            "test_ensemble_stacking_logistic_5.png",
        ),
        (
            "prefit",
            np.array([0.0, 0.0, 0.66666667, 0.66666667, 1.0, 1.0, 1.0, 1.0]),
            np.array(
                [
                    0.0,
                    0.14285714,
                    0.14285714,
                    0.28571429,
                    0.28571429,
                    0.57142857,
                    0.85714286,
                    1.0,
                ]
            ),
            np.array(
                [
                    0.43149058,
                    0.42126726,
                    0.47809721,
                    0.52508833,
                    0.47288467,
                    0.46767804,
                    0.55622149,
                    0.57674934,
                    0.43149058,
                    0.53549986,
                ]
            ),
            "test_ensemble_stacking_logistic_prefit.png",
        ),
    ],
)
def test_ensemble_stacking_logistic(tmpdir, cv, exp_fpr, exp_tpr, exp_pred, exp_plot):
    expected_plot = files("viral_seq.tests.expected").joinpath(exp_plot)
    rng = np.random.default_rng(seed=2025)
    n_samples = 10
    X_train = pd.DataFrame(
        rng.random(size=(n_samples, 10)), columns=[f"Feature {i}" for i in range(10)]
    )
    y_train = rng.choice(2, n_samples)
    X_test = pd.DataFrame(
        rng.random(size=(n_samples, 10)), columns=[f"Feature {i}" for i in range(10)]
    )
    y_test = rng.choice(2, n_samples)
    models = []
    for i in range(2):
        this_model = RandomForestClassifier(random_state=2025)
        if cv == "prefit":
            this_model.fit(X_train, y_train)
        models.append((f"RF {i}", this_model))
    test_file = tmpdir / "test_file.csv"
    pd.DataFrame({"Species": [f"Species {i}" for i in range(n_samples)]}).to_csv(
        test_file
    )
    predictions_path = tmpdir
    plots_path = tmpdir
    estimator_names = [f"Name {i}" for i in range(2)]
    fpr, tpr = workflow._ensemble_stacking_logistic(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        test_file,
        predictions_path,
        plots_path,
        cv=cv,
        plot_weights=True,
        estimator_names=estimator_names,
    )
    pred = pd.read_csv(
        predictions_path / f"StackingClassifier_LR_predictions_cv_{cv}.csv"
    )[f"StackingClassifier LR CV={cv}"].values
    assert_allclose(fpr, exp_fpr)
    assert_allclose(tpr, exp_tpr)
    assert_allclose(pred, exp_pred)
    assert (
        compare_images(
            expected_plot, tmpdir / f"StackingClassifier_LR_weights_cv_{cv}.png", 0.001
        )
        is None
    )


def test_ensemble_stacking_logistic_value_error(tmpdir):
    rng = np.random.default_rng(seed=2025)
    X_train = pd.DataFrame(
        np.zeros((10, 10)), columns=[f"Feature {i}" for i in range(10)]
    )
    y_train = rng.choice(2, 10)
    X_test = X_train
    y_test = y_train
    models = [("RF", RandomForestClassifier().fit(X_train, y_train))]
    test_file = tmpdir / "test_file.csv"
    pd.DataFrame({"Species": [f"Species {i}" for i in range(10)]}).to_csv(test_file)
    predictions_path = tmpdir
    plots_path = tmpdir
    with pytest.raises(ValueError, match="estimator_names"):
        workflow._ensemble_stacking_logistic(
            models,
            X_train,
            y_train,
            X_test,
            y_test,
            test_file,
            predictions_path,
            plots_path,
            cv="prefit",
            plot_weights=True,
            estimator_names=None,
        )
