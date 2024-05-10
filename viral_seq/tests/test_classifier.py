from sklearn.datasets import make_classification
from viral_seq.analysis import rfc_utils, classifier
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from hypothesis import given, strategies as st


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
        rfc_utils,
        X,
        y,
        n_splits=3,
        n_estimators=10,
        random_state=random_state,
    )
    assert_allclose(test_score, score)


@given(
    max_depth=st.floats(
        min_value=0.0,
        max_value=9.223372036854776e18,
        exclude_max=True,
        allow_nan=False,
        allow_infinity=False,
    ),  # max_value set to C long max, max accepted by numpy
    criterion=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    class_weight=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    floats=st.booleans(),
)
def test_cv_score_args(max_depth, criterion, class_weight, floats):
    X, y = make_classification(n_samples=10, n_features=10)
    X = pd.DataFrame(X)
    if not floats:
        max_depth = rfc_utils._floatparam("max_depth", max_depth)
        criterion = rfc_utils._floatparam("criterion", criterion)
        class_weight = rfc_utils._floatparam("class_weight", class_weight)
    classifier.cv_score(
        rfc_utils,
        X,
        y,
        n_splits=3,
        max_depth=max_depth,
        criterion=criterion,
        class_weight=class_weight,
        n_estimators=10,
    )


@pytest.mark.parametrize(
    "max_depth, criterion, class_weight",
    [
        ("bad value", "gini", None),
        (1, "bad value", None),
        (1, "gini", "bad value"),
    ],
)
def test_cv_score_args_passed(max_depth, criterion, class_weight):
    X, y = make_classification(n_samples=10, n_features=10)
    X = pd.DataFrame(X)
    with pytest.raises(ValueError, match="Got 'bad value' instead."):
        classifier.cv_score(
            rfc_utils,
            X,
            y,
            n_splits=3,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            n_estimators=10,
        )


@pytest.mark.parametrize(
    "distributions, score, params",
    [
        (
            {
                "max_samples": (0.1, 1.0),
                "min_samples_leaf": (0.1, 1.0),
                "min_samples_split": (0.1, 1.0),
                "max_features": (0.1, 0.6325),
                "criterion": (0.0, 1.0),
                "class_weight": (0.0, 1.0),
                "max_depth": (0.0, 30.99999),
            },
            1.0,
            {
                "class_weight": None,
                "criterion": "gini",
                "max_depth": None,
                "max_features": 0.31622776601683794,
                "max_samples": 1.0,
                "min_samples_leaf": 0.1,
                "min_samples_split": 0.2,
            },
        ),
        ({"max_samples": (0.0, 0.5)}, 1.0, {"max_samples": 1.0}),
    ],
)
def test_get_hyperparameters(distributions, score, params):
    X, y = make_classification(n_samples=10, n_features=10, random_state=0)
    X = pd.DataFrame(X)
    res = classifier.get_hyperparameters(
        model_utils=rfc_utils,
        X=X,
        y=y,
        bayes_parameters={
            "init_points": 1,
            "n_iter": 1,
        },
        model_parameters={
            "n_estimators": 10,
            "n_jobs": 1,
        },
        random_state=0,
        distributions=distributions,
    )
    assert res["target"] == pytest.approx(score)
    assert_array_equal(
        np.sort(list(res["params"].keys())), np.sort(list(params.keys()))
    )
    for key in res["params"].keys():
        assert res["params"][key] == pytest.approx(params[key])


def test_get_hyperparameters_duplicate_params():
    X, y = make_classification(n_samples=10, n_features=10, random_state=0)
    X = pd.DataFrame(X)
    with pytest.raises(ValueError, match="max_depth"):
        classifier.get_hyperparameters(
            model_utils=rfc_utils,
            X=X,
            y=y,
            model_parameters={
                "max_depth": 1,
            },
            random_state=0,
            distributions={
                "max_depth": (0.0, 10.0),
            },
        )
