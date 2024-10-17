from sklearn.datasets import make_classification
from viral_seq.analysis import classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal
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


@pytest.mark.parametrize("model, name", [(RandomForestClassifier, "rfc")])
def test_train_and_predict(capsys, model, name, tmpdir):
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
            model_out="model.p",
            params_predict=params_predict,
            params_optimized=params_optimized,
        )
        assert Path("model.p").is_file()
        captured = capsys.readouterr()
        assert (
            captured.out
            == "Will train model and run prediction on test for Classifier using the following parameters:\n{'random_state': 0, 'verbose': 0}\nSaving trained model to model.p\n"
        )
        df = pd.DataFrame(y_pred)
        y_expected = files("viral_seq.tests.expected") / (
            f"test_train_and_predict_y_expected_{name}.csv"
        )
        df_expected = pd.read_csv(y_expected)
        df_expected.columns = df_expected.columns.astype(int)
        assert_frame_equal(df, df_expected)
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
