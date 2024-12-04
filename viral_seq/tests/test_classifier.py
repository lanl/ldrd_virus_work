from sklearn.datasets import make_classification
from viral_seq.analysis import classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from ray import tune
from pathlib import Path
from importlib.resources import files
from sklearn.utils.validation import check_is_fitted
from matplotlib.testing.compare import compare_images


@pytest.mark.parametrize(
    "random_state, score",
    [
        (3, 0.625),
        (4, 1.0),
        (5, 0.875),
    ],
)
def test_cv_score(random_state, score):
    X, y = make_classification(n_samples=10, n_features=10, random_state=random_state)
    X = pd.DataFrame(X)
    test_score = classifier.cv_score(
        RandomForestClassifier,
        X,
        y,
        n_splits=3,
        n_estimators=10,
        random_state=random_state,
    )
    assert_allclose(test_score, score)


# even running this test with num_samples=2 takes > 10 s and is not very useful
@pytest.mark.slow
@pytest.mark.parametrize(
    "config, score, best_params",
    [
        (
            {
                "n_estimators": 10,
                "n_jobs": 1,
                "max_samples": tune.uniform(0.1, 1.0),
                "min_samples_leaf": tune.uniform(0.1, 1.0),
                "min_samples_split": tune.uniform(0.1, 1.0),
                "max_features": tune.uniform(0.1, 0.6325),
                "criterion": tune.choice(["gini", "log_loss"]),
                "class_weight": tune.choice([None, "balanced", "balanced_subsample"]),
                "max_depth": tune.choice([None] + [i for i in range(1, 31)]),
            },
            1.0,
            {
                "n_estimators": 10,
                "n_jobs": 1,
                "max_samples": 0.8774700453309082,
                "min_samples_leaf": 0.20577867036582978,
                "min_samples_split": 0.5656411964387028,
                "max_features": 0.17032626662879413,
                "criterion": "gini",
                "class_weight": None,
                "max_depth": 21,
            },
        ),
        (
            {
                "n_estimators": 10,
                "n_jobs": 1,
                "max_samples": tune.uniform(0.0, 0.2),
                "max_features": tune.uniform(0.1, 0.2),
            },
            1.0,
            {
                "n_estimators": 10,
                "n_jobs": 1,
                "max_samples": 0.19273255210020587,
                "max_features": 0.1383441518825778,
            },
        ),
    ],
)
def test_get_hyperparameters(config, score, best_params):
    X, y = make_classification(n_samples=10, n_features=10, random_state=0)
    X = pd.DataFrame(X)
    res = classifier.get_hyperparameters(
        model=RandomForestClassifier,
        X=X,
        y=y,
        num_samples=10,
        config=config,
        random_state=0,
        max_concurrent_trials=0,
    )
    assert res["target"] == pytest.approx(score)
    assert_array_equal(
        np.sort(list(res["params"].keys())), np.sort(list(best_params.keys()))
    )
    for key in res["params"].keys():
        assert res["params"][key] == pytest.approx(best_params[key])


@pytest.mark.parametrize(
    "model, name, calibrate, expected_values",
    [
        (RandomForestClassifier, "rfc", False, [0.4, 0.31, 0.26, 0.47, 0.64]),
        (
            RandomForestClassifier,
            "rfc",
            True,
            [
                0.4624745376650326,
                0.3246510677646521,
                0.485242552665174,
                0.519451223839514,
                0.5599988315132807,
            ],
        ),
    ],
)
def test_train_and_predict(capsys, model, name, calibrate, expected_values, tmpdir):
    X, y = make_classification(n_samples=15, n_features=10, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=5, random_state=0
    )
    params_predict = {"random_state": 0}
    params_optimized = {"random_state": 42, "verbose": 0}
    with tmpdir.as_cwd():
        clf, y_pred = classifier.train_and_predict(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_out="model.p",
            params_predict=params_predict,
            params_optimized=params_optimized,
            calibrate=calibrate,
        )
        assert Path("model.p").is_file()
        captured = capsys.readouterr()
        if calibrate:
            assert (
                captured.out
                == "Will train model and run prediction on test for Classifier using the following parameters:\n{'random_state': 0, 'verbose': 0}\nClassifier achieved ROC AUC = 1.00 on test data.\nCalibrating model with cross-validation\nClassifier achieved ROC AUC = 0.50 on test data after calibration with cross-validation.\nSaving trained model to model.p\n"
            )
        else:
            assert (
                captured.out
                == "Will train model and run prediction on test for Classifier using the following parameters:\n{'random_state': 0, 'verbose': 0}\nClassifier achieved ROC AUC = 1.00 on test data.\nSaving trained model to model.p\n"
            )
        df = pd.DataFrame(y_pred)
        assert_allclose(df[0].values, expected_values)
        check_is_fitted(clf)


def test_get_model_args():
    actual = classifier.get_model_arguments(
        n_jobs=1, random_state=1, num_samples=10, num_features=10
    )
    expected_keys = {"model", "group", "suffix", "optimize", "predict"}
    for key, subdict in actual.items():
        assert set(subdict.keys()) == expected_keys


def test_get_roc_curve():
    rng = np.random.default_rng(123)
    y_true = rng.integers(2, size=10)
    predictions = {}
    predictions["Test1"] = rng.random((10,))
    predictions["Test2"] = rng.random((10,))
    fpr, tpr, tpr_std = classifier.get_roc_curve(y_true, predictions)
    expected_fpr = np.linspace(0, 1, 100)
    expected_tpr = np.ones(100)
    expected_tpr[0] = 0
    expected_tpr[1:43] = 0.5
    expected_tpr[43:57] = 2 / 3
    expected_tpr[57:71] = 5 / 6
    expected_tpr_std = np.zeros(100)
    expected_tpr_std[1:43] = 1 / 6
    expected_tpr_std[57:71] = 1 / 6
    assert_allclose(fpr, expected_fpr)
    assert_allclose(tpr, expected_tpr)
    assert_allclose(tpr_std, expected_tpr_std)


def test_plot_roc_curve(tmpdir):
    rng = np.random.default_rng(951753)
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_roc_curve.png"
    )
    # generate random input typical of fpr, tpr values
    # arrays of values increasing by random intervals from 0.0 to 1.0
    tpr = np.sort(np.append(rng.random((8,)), [0.0, 1.0]))
    fpr = np.sort(np.append(rng.random((8,)), [0.0, 1.0]))
    with tmpdir.as_cwd():
        classifier.plot_roc_curve("Test", fpr, tpr)
        assert compare_images(expected_plot, "roc_plot.png", 0.001) is None


def test_plot_roc_curve_comparison(tmpdir):
    rng = np.random.default_rng(654987)
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_roc_curve_comparison.png"
    )
    # generate random input typical of fpr, tpr, and tpr_std values
    # list of two arrays; each of values increasing by random intervals from 0.0 to 1.0
    fprs = [np.sort(np.append(rng.random((8,)), [0.0, 1.0]))] + [
        np.sort(np.append(rng.random((8,)), [0.0, 1.0]))
    ]
    tprs = [np.sort(np.append(rng.random((8,)), [0.0, 1.0]))] + [
        np.sort(np.append(rng.random((8,)), [0.0, 1.0]))
    ]
    # list of two arrays; each beginning and ending with 0.0, with middle values in the range [0.0, 0.1)
    tpr_stds = [np.append(np.append([0.0], rng.uniform(0.0, 0.1, (8,))), [0.0])] + [
        np.append(np.append([0.0], rng.uniform(0.0, 0.1, (8,))), [0.0])
    ]
    with tmpdir.as_cwd():
        classifier.plot_roc_curve_comparison(["Test1", "Test2"], fprs, tprs, tpr_stds)
        assert compare_images(expected_plot, "roc_plot_comparison.png", 0.001) is None


@pytest.mark.parametrize(
    "random_state, y_tests, y_preds, scores",
    [
        (
            123456,
            [np.array([0, 1, 1, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])],
            [
                np.array([0.2, 0.6, 0.5, 0.6]),
                np.array([0.4, 0.5, 0.3]),
                np.array([0.3, 0.3, 0.3]),
            ],
            [0.625, 0.5, 0.5],
        ),
        (
            159758,
            [np.array([1, 1, 0, 0]), np.array([1, 0, 1]), np.array([1, 0, 0])],
            [
                np.array([0.8, 0.8, 0.2, 0.2]),
                np.array([0.8, 0.4, 0.6]),
                np.array([0.7, 0.1, 0.2]),
            ],
            [1.0, 1.0, 1.0],
        ),
        (
            654987,
            [np.array([1, 0, 1, 0]), np.array([0, 1, 1]), np.array([0, 0, 1])],
            [
                np.array([0.9, 0.1, 0.6, 0.5]),
                np.array([0.1, 0.8, 0.9]),
                np.array([0.3, 0.3, 1.0]),
            ],
            [1.0, 1.0, 1.0],
        ),
    ],
)
def test_cross_validation(random_state, y_tests, y_preds, scores):
    X, y = make_classification(n_samples=10, n_features=10, random_state=random_state)
    X = pd.DataFrame(X)
    test_y_tests, test_y_preds, test_scores = classifier.cross_validation(
        RandomForestClassifier,
        X,
        y,
        n_splits=3,
        n_estimators=10,
        random_state=random_state,
    )
    assert_allclose(np.hstack(test_y_tests), np.hstack(y_tests))
    assert_allclose(np.hstack(test_y_preds), np.hstack(y_preds))
    assert_allclose(test_scores, scores)


def test_get_roc_curve_cv():
    rng = np.random.default_rng(456987)
    folds = 3
    copies = 1
    # different folds sometimes have different numbers of samples
    samples = rng.integers(5, 7, size=folds)
    # `append` 1 to ensure tprs can be calculated
    y_tests = [np.append(rng.integers(2, size=samples[i] - 1), 1) for i in range(folds)]
    predictions = [
        [rng.random(size=samples[i]) for i in range(folds)] for _ in range(copies)
    ]
    cv_roc_data = classifier.get_roc_curve_cv(y_tests, predictions)
    exp_mean_tpr = np.ones(100)
    exp_mean_tpr[0] = 0.0
    exp_mean_tpr[1:-1] = 4.0 / 9.0
    exp_tpr_std = np.zeros(100)
    exp_tpr_std[1:-1] = 0.41573971
    exp_fpr_folds = [
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]
    exp_tpr_folds = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.33333333, 0.33333333, 1.0]),
        np.array([0.0, 0.25, 1.0, 1.0]),
    ]
    assert_allclose(cv_roc_data.mean_tpr, exp_mean_tpr)
    assert_allclose(cv_roc_data.tpr_std, exp_tpr_std)
    assert_allclose(np.hstack(cv_roc_data.fpr_folds), np.hstack(exp_fpr_folds))
    assert_allclose(np.hstack(cv_roc_data.tpr_folds), np.hstack(exp_tpr_folds))
    assert cv_roc_data.tpr_std_folds is None


# using pytest parametrize seemed about as verbose as just making two tests
def test_get_roc_curve_cv_copies():
    rng = np.random.default_rng(13579)
    folds = 3
    copies = 5
    # different folds sometimes have different numbers of samples
    samples = rng.integers(5, 7, size=folds)
    # `append` 1 to ensure tprs can be calculated
    y_tests = [np.append(rng.integers(2, size=samples[i] - 1), 1) for i in range(folds)]
    predictions = [
        [rng.random(size=samples[i]) for i in range(folds)] for _ in range(copies)
    ]
    cv_roc_data = classifier.get_roc_curve_cv(y_tests, predictions)
    exp_mean_tpr = np.ones(100)
    exp_mean_tpr[0] = 0.0
    exp_mean_tpr[1:25] = 1.0 / 3.0
    exp_mean_tpr[25:50] = 0.4
    exp_mean_tpr[50:-1] = 19.0 / 30.0
    exp_tpr_std = np.zeros(100)
    exp_tpr_std[1:25] = 0.40483193
    exp_tpr_std[25:50] = 0.42622373
    exp_tpr_std[50:-1] = 0.38586123
    exp_tpr_folds = [[] for _ in range(folds)]
    exp_tpr_folds[0] = np.ones(100)
    exp_tpr_folds[0][0] = 0.0
    exp_tpr_folds[0][1:50] = 0.3
    exp_tpr_folds[0][50:-1] = 0.75
    exp_tpr_folds[1] = np.ones(100)
    exp_tpr_folds[1][0] = 0.0
    exp_tpr_folds[1][1:25] = 0.4
    exp_tpr_folds[1][25:-1] = 0.6
    exp_tpr_folds[2] = np.ones(100)
    exp_tpr_folds[2][0] = 0.0
    exp_tpr_folds[2][1:50] = 0.3
    exp_tpr_folds[2][50:-1] = 0.55
    exp_tpr_std_folds = [[] for _ in range(folds)]
    exp_tpr_std_folds[0] = np.zeros(100)
    exp_tpr_std_folds[0][1:50] = 0.29154759
    exp_tpr_std_folds[0][50:-1] = 0.2236068
    exp_tpr_std_folds[1] = np.zeros(100)
    exp_tpr_std_folds[1][1:-1] = 0.48989795
    exp_tpr_std_folds[2] = np.zeros(100)
    exp_tpr_std_folds[2][1:50] = 0.4
    exp_tpr_std_folds[2][50:-1] = 0.36742346
    assert_allclose(cv_roc_data.mean_tpr, exp_mean_tpr)
    assert_allclose(cv_roc_data.tpr_std, exp_tpr_std)
    assert cv_roc_data.fpr_folds is None
    assert_allclose(cv_roc_data.tpr_folds, exp_tpr_folds)
    assert_allclose(cv_roc_data.tpr_std_folds, exp_tpr_std_folds)


def test_plot_calibration_curve(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_calibration_curve.png"
    )
    rng = np.random.default_rng(123)
    y_test = rng.integers(2, size=10)
    y_pred = rng.random(size=10)
    y_pred_calibrated = rng.random(size=10)
    with tmpdir.as_cwd():
        classifier.plot_calibration_curve(y_test, y_pred, y_pred_calibrated)
        assert (
            compare_images(expected_plot, "plot_calibration_curve.png", 0.001) is None
        )


def test_plot_calibration_curves(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_calibration_curves.png"
    )
    rng = np.random.default_rng(123)
    predictions = {}
    for i in range(5):
        predictions[str(i)] = rng.random(size=10)
    y_test = rng.integers(2, size=10)
    with tmpdir.as_cwd():
        classifier.plot_calibration_curves(y_test, predictions)
        assert (
            compare_images(expected_plot, "plot_calibration_curve.png", 0.001) is None
        )
