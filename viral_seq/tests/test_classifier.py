from sklearn.datasets import make_classification
from viral_seq.analysis import classifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from hypothesis import given, example, strategies as st
from hypothesis.extra import numpy as hnp
import math


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
    "model, name, calibrate, rtol, expected_msg, expected_values",
    [
        (
            RandomForestClassifier,
            "rfc",
            False,
            1e-7,
            "Will train model and run prediction on test for Classifier using the following parameters:\n{'random_state': 0, 'verbose': 0}\nClassifier achieved ROC AUC = 1.00 on test data.\nSaving trained model to model.p\n",
            [0.4, 0.31, 0.26, 0.47, 0.64],
        ),
        (
            RandomForestClassifier,
            "rfc",
            True,
            1e-5,  # tolerance lowered due to CalibratedClassifierCV instability across versions
            "Will train model and run prediction on test for Classifier using the following parameters:\n{'random_state': 0, 'verbose': 0}\nClassifier achieved ROC AUC = 1.00 on test data.\nCalibrating model with cross-validation\nClassifier achieved ROC AUC = 0.50 on test data after calibration with cross-validation.\nSaving trained model to model.p\n",
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
def test_train_and_predict(
    capsys, model, name, calibrate, rtol, expected_msg, expected_values, tmpdir
):
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
        assert captured.out == expected_msg
        df = pd.DataFrame(y_pred)
        assert_allclose(df[0].values, expected_values, rtol=rtol)
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


@pytest.mark.parametrize(
    "get_eer_data, seed, expected_filename",
    [
        (False, 951753, "test_plot_roc_curve.png"),
        (
            True,
            951753,
            "test_plot_roc_curve_eer_horz.png",
        ),
        (
            True,
            357687,
            "test_plot_roc_curve_eer_vert.png",
        ),
    ],
)
def test_plot_roc_curve(get_eer_data, seed, expected_filename, tmpdir):
    rng = np.random.default_rng(seed)
    expected_plot = files("viral_seq.tests.expected").joinpath(expected_filename)
    # generate random input typical of fpr, tpr values
    # arrays of values increasing by random intervals from 0.0 to 1.0
    # values are repeated to generate typical stepped curve
    tpr = np.repeat(np.sort(np.append(rng.random((4,)), [0.0, 1.0])), 2)[0:-1]
    fpr = np.repeat(np.sort(np.append(rng.random((4,)), [0.0, 1.0])), 2)[1:]
    threshold = np.sort(np.append(rng.random((9,)), [0.0, 1.0]))
    if get_eer_data:
        eer_data = classifier.cal_eer_thresh_and_val(fpr, tpr, threshold)
    else:
        eer_data = None
    with tmpdir.as_cwd():
        classifier.plot_roc_curve("Test", fpr, tpr, eer_data=eer_data)
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
    "random_state, y_tests, y_preds, scores, y_proba",
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
            [0.2, 0.6, 0.5, 0.4, 0.6, 0.3, 0.5, 0.3, 0.3, 0.3],
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
            [0.8, 0.8, 0.2, 0.2, 0.8, 0.4, 0.6, 0.7, 0.1, 0.2],
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
            [0.9, 0.1, 0.6, 0.5, 0.1, 0.3, 0.8, 0.9, 0.3, 1.0],
        ),
    ],
)
def test_cross_validation(random_state, y_tests, y_preds, scores, y_proba):
    X, y = make_classification(n_samples=10, n_features=10, random_state=random_state)
    X = pd.DataFrame(X)
    cv_data = classifier.cross_validation(
        RandomForestClassifier,
        X,
        y,
        n_splits=3,
        n_estimators=10,
        random_state=random_state,
    )
    assert_allclose(np.hstack(cv_data.y_tests), np.hstack(y_tests))
    assert_allclose(np.hstack(cv_data.y_preds), np.hstack(y_preds))
    assert_allclose(cv_data.scores, scores)
    assert_allclose(cv_data.y_proba, y_proba)


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


def test_cal_eer_thresh_and_val():
    """Regression test calc_eer_thresh_and_val."""
    rng = np.random.default_rng(4684365)
    expected_eer_data = classifier.EER_Data(
        eer_threshold_index=5,
        eer_threshold=0.6575733297477135,
        eer_value=0.4434414483363267,
    )
    tpr = np.sort(np.append(rng.random((8,)), [0.0, 1.0]))
    fpr = np.sort(np.append(rng.random((8,)), [0.0, 1.0]))
    threshold = np.sort(np.append(rng.random((8,)), [0.0, 1.0]))
    eer_data = classifier.cal_eer_thresh_and_val(fpr, tpr, threshold)
    assert_allclose(eer_data, expected_eer_data)


@pytest.mark.parametrize(
    "input_data, expected_eer_data",
    [
        (np.arange(0, 1, 0.1), (5, 0.5, 0.5)),
        (np.arange(0, 1, 0.01), (50, 0.5, 0.5)),
    ],
)
def test_cal_eer_thresh_and_val_functional(input_data, expected_eer_data):
    """Check expected test cases of calc_eer_thresh_and_val."""
    tpr = input_data
    fpr = input_data
    threshold = input_data
    eer_data = classifier.cal_eer_thresh_and_val(fpr, tpr, threshold)
    assert_allclose(eer_data, expected_eer_data)


def test_plot_confusion_matrix(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_confusion_matrix.png"
    )
    rng = np.random.default_rng(123)
    y_test = rng.integers(2, size=10)
    y_pred = rng.integers(2, size=10)
    with tmpdir.as_cwd():
        classifier.plot_confusion_matrix(y_test, y_pred)
        assert compare_images(expected_plot, "confusion_matrix.png", 0.001) is None


def test_plot_confusion_matrix_mean(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_confusion_matrix_mean.png"
    )
    rng = np.random.default_rng(123)
    y_test = rng.integers(2, size=10)
    y_preds = [rng.integers(2, size=10) for i in range(3)]
    with tmpdir.as_cwd():
        classifier.plot_confusion_matrix_mean(y_test, y_preds)
        assert compare_images(expected_plot, "confusion_matrix_mean.png", 0.001) is None


def test_plot_logistic_stacked_weights(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_logistic_stacked_weights.png"
    )
    rng = np.random.default_rng(2025)
    X = rng.random(size=(10, 10))
    y = rng.choice(2, 10)
    models = [
        (f"RF {i}", RandomForestClassifier(random_state=i).fit(X, y)) for i in range(2)
    ]
    stacked_logistic = StackingClassifier(models, cv="prefit").fit(X, y)
    estimator_names = [f"Name {i}" for i in range(2)]
    file_name = "Test.png"
    with tmpdir.as_cwd():
        classifier._plot_logistic_stacked_weights(
            stacked_logistic, estimator_names, file_name
        )
        assert compare_images(expected_plot, "Test.png", 0.001) is None


@pytest.mark.parametrize(
    "cv, exp_fpr, exp_tpr, exp_proba, exp_plot",
    [
        (
            5,
            np.array([0.0] * 2 + [2 / 3] * 2 + [1.0] * 2),
            np.array([0.0, 1 / 7, 2 / 7, 5 / 7, 6 / 7, 1.0]),
            np.array(
                [
                    0.42008643,
                    0.42008643,
                    0.42021615,
                    0.28240654,
                    0.44251979,
                    0.42008643,
                    0.28240654,
                    0.42021615,
                    0.28229865,
                    0.42021615,
                ]
            ),
            "test_ensemble_stacking_logistic_5.png",
        ),
        (
            "prefit",
            np.array([0.0] + [1 / 3] * 2 + [1.0] * 3),
            np.array([0.0, 1 / 7, 2 / 7, 3 / 7, 6 / 7, 1.0]),
            np.array(
                [
                    0.64180405,
                    0.64180405,
                    0.64398981,
                    0.7341501,
                    0.37633148,
                    0.64180405,
                    0.7341501,
                    0.64398981,
                    0.73228778,
                    0.64398981,
                ]
            ),
            "test_ensemble_stacking_logistic_prefit.png",
        ),
    ],
)
def test_ensemble_stacking_logistic(tmpdir, cv, exp_fpr, exp_tpr, exp_proba, exp_plot):
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
    for i, model in enumerate(
        [RandomForestClassifier, ExtraTreesClassifier, LGBMClassifier, XGBClassifier]
    ):
        this_model = model(random_state=2025, n_estimators=1)
        if cv == "prefit":
            this_model.fit(X_train, y_train)
        models.append((f"Model {i}", this_model))
    test_file = tmpdir / "test_file.csv"
    pd.DataFrame({"Species": [f"Species {i}" for i in range(n_samples)]}).to_csv(
        test_file
    )
    predictions_path = tmpdir
    plots_path = tmpdir
    estimator_names = [f"Name {i}" for i in range(len(models))]
    pred, fpr, tpr = classifier._ensemble_stacking_logistic(
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
    proba = pd.read_csv(
        predictions_path / f"StackingClassifier_LR_predictions_cv_{cv}.csv"
    )[f"StackingClassifier LR CV={cv}"].values
    assert_allclose(pred, exp_proba > 0.5)
    assert_allclose(fpr, exp_fpr)
    assert_allclose(tpr, exp_tpr)
    assert_allclose(proba, exp_proba)
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
        classifier._ensemble_stacking_logistic(
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


def test_roc_intersection_bug(tmpdir):
    fprs = [np.array([0.0, 0.17, 0.17, 0.2, 0.2, 1.0])]
    tprs = [np.array([0.0, 0.0, 0.8, 0.8, 1.0, 1.0])]
    eer_data = classifier.EER_Data(3, 0, 0)
    with tmpdir.as_cwd():
        # bug will cause AttributeError: 'LineString' object has no attribute 'x'
        classifier.plot_roc_curve_comparison(
            ["Test"], fprs, tprs, eer_data_list=[eer_data]
        )


@pytest.mark.parametrize(
    "y_test, expected_entropy",
    [
        (np.append(np.zeros(1), np.ones(9)), 0.5746356978376793),
        (np.append(np.zeros(2), np.ones(8)), 0.7732266742876346),
        (np.append(np.zeros(3), np.ones(7)), 0.9023932827949788),
        (np.append(np.zeros(4), np.ones(6)), 0.9760206482366149),
        (np.append(np.zeros(5), np.ones(5)), 1.0),
    ],
)
def test_entropy(y_test, expected_entropy):
    """Regression test entropy values."""
    entropy = classifier.entropy(y_test)
    assert entropy == pytest.approx(expected_entropy)


@given(
    hnp.arrays(
        dtype=np.int32,
        elements=st.sampled_from([0, 1]),
        shape=st.integers(min_value=10, max_value=2_000),
    ),
)
@example(np.zeros(10))
@example(np.ones(10))
def test_entropy_not_zero_or_nan(y_test):
    """Protect against nan or zero values for entropy which break the workflow."""
    entropy = classifier.entropy(y_test)
    assert entropy != 0.0 and not math.isnan(entropy)


def test_ensemble_entropy(tmpdir):
    rng = np.random.default_rng(seed=2025)
    exp_fpr = np.array([0.0] + 2 * [1 / 3] + 3 * [1])
    exp_tpr = np.array([0.0, 1 / 7, 2 / 7, 3 / 7, 6 / 7, 1.0])
    exp_proba = np.array(
        [
            0.52522748,
            0.52522748,
            0.55593207,
            0.83657937,
            0.44406794,
            0.52522748,
            0.83657937,
            0.55593207,
            0.80587477,
            0.55593207,
        ]
    )
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
    for i, model in enumerate(
        [RandomForestClassifier, ExtraTreesClassifier, LGBMClassifier, XGBClassifier]
    ):
        this_model = model(random_state=2025, n_estimators=1)
        this_model.fit(X_train, y_train)
        models.append((f"Model {i}", this_model))
    weights = list(rng.random(len(models)))
    test_file = tmpdir / "test_file.csv"
    pd.DataFrame({"Species": [f"Species {i}" for i in range(n_samples)]}).to_csv(
        test_file
    )
    predictions_path = tmpdir
    pred, fpr, tpr = classifier._ensemble_entropy(
        models,
        weights,
        X_train,
        y_train,
        X_test,
        y_test,
        test_file,
        predictions_path,
    )
    proba = pd.read_csv(predictions_path / "Entropy_predictions.csv")[
        "Entropy Weighted Ensemble"
    ].values
    assert_allclose(pred, exp_proba > 0.5)
    assert_allclose(fpr, exp_fpr)
    assert_allclose(tpr, exp_tpr)
    assert_allclose(proba, exp_proba)


@given(
    y_test=hnp.arrays(
        dtype=np.int32,
        elements=st.sampled_from([0, 1]),
        shape=st.integers(min_value=10, max_value=2_000),
    ),
    random_state=hnp.from_dtype(np.dtype(np.uint32)),
)
def test_entropy_func_order(y_test, random_state):
    """Test property that input order does not affect entropy."""
    ent = classifier.entropy(y_test)
    rng = np.random.default_rng(random_state)
    y_shuff = rng.permutation(y_test)
    ent_shuff = classifier.entropy(y_shuff)
    assert ent == pytest.approx(ent_shuff)


@given(
    hnp.arrays(
        dtype=np.int32,
        elements=st.sampled_from([0, 1]),
        shape=st.integers(min_value=10, max_value=2_000),
    ),
)
def test_entropy_func_invert(y_test):
    """Test property that inverse of input has same entropy."""
    ent = classifier.entropy(y_test)
    y_inv = 1 - y_test
    ent_inv = classifier.entropy(y_inv)
    assert ent == pytest.approx(ent_inv)


@given(
    random_state=hnp.from_dtype(np.dtype(np.uint32)),
    half_size=st.integers(min_value=5, max_value=1_000),
)
def test_entropy_func_equal(random_state, half_size):
    """Test property that equal number of each class has entropy 1."""
    rng = np.random.default_rng(random_state)
    y_test = np.array([0] * half_size + [1] * half_size)
    rng.shuffle(y_test)
    entropy = classifier.entropy(y_test)
    assert entropy == 1.0
