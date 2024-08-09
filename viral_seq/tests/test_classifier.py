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
    expected_fpr = np.array(
        [
            0.0,
            0.010101010101010102,
            0.020202020202020204,
            0.030303030303030304,
            0.04040404040404041,
            0.05050505050505051,
            0.06060606060606061,
            0.07070707070707072,
            0.08080808080808081,
            0.09090909090909091,
            0.10101010101010102,
            0.11111111111111112,
            0.12121212121212122,
            0.13131313131313133,
            0.14141414141414144,
            0.15151515151515152,
            0.16161616161616163,
            0.17171717171717174,
            0.18181818181818182,
            0.19191919191919193,
            0.20202020202020204,
            0.21212121212121213,
            0.22222222222222224,
            0.23232323232323235,
            0.24242424242424243,
            0.25252525252525254,
            0.26262626262626265,
            0.27272727272727276,
            0.2828282828282829,
            0.29292929292929293,
            0.30303030303030304,
            0.31313131313131315,
            0.32323232323232326,
            0.33333333333333337,
            0.3434343434343435,
            0.3535353535353536,
            0.36363636363636365,
            0.37373737373737376,
            0.38383838383838387,
            0.393939393939394,
            0.4040404040404041,
            0.4141414141414142,
            0.42424242424242425,
            0.43434343434343436,
            0.4444444444444445,
            0.4545454545454546,
            0.4646464646464647,
            0.4747474747474748,
            0.48484848484848486,
            0.494949494949495,
            0.5050505050505051,
            0.5151515151515152,
            0.5252525252525253,
            0.5353535353535354,
            0.5454545454545455,
            0.5555555555555556,
            0.5656565656565657,
            0.5757575757575758,
            0.5858585858585859,
            0.595959595959596,
            0.6060606060606061,
            0.6161616161616162,
            0.6262626262626263,
            0.6363636363636365,
            0.6464646464646465,
            0.6565656565656566,
            0.6666666666666667,
            0.6767676767676768,
            0.686868686868687,
            0.696969696969697,
            0.7070707070707072,
            0.7171717171717172,
            0.7272727272727273,
            0.7373737373737375,
            0.7474747474747475,
            0.7575757575757577,
            0.7676767676767677,
            0.7777777777777778,
            0.787878787878788,
            0.797979797979798,
            0.8080808080808082,
            0.8181818181818182,
            0.8282828282828284,
            0.8383838383838385,
            0.8484848484848485,
            0.8585858585858587,
            0.8686868686868687,
            0.8787878787878789,
            0.888888888888889,
            0.8989898989898991,
            0.9090909090909092,
            0.9191919191919192,
            0.9292929292929294,
            0.9393939393939394,
            0.9494949494949496,
            0.9595959595959597,
            0.9696969696969697,
            0.9797979797979799,
            0.98989898989899,
            1.0,
        ]
    )
    expected_tpr = np.array(
        [
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            0.8333333333333333,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    expected_tpr_std = np.array(
        [
            0.0,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.16666666666666669,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
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
