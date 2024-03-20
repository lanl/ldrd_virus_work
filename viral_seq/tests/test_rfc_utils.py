from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from viral_seq.analysis import rfc_utils
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from hypothesis import example, given, strategies as st


@pytest.mark.parametrize(
    "metric",
    (accuracy_score, balanced_accuracy_score),
)
@pytest.mark.parametrize(
    "class_weights",
    (None, [0.2, 0.8]),
)
def test_oob_score_raw_pred(metric, class_weights):
    X, y = make_classification(
        n_samples=10, n_features=10, weights=class_weights, random_state=0
    )
    X = pd.DataFrame(X)
    # we can check accuracy_score against built-in sklearn oob_score
    clf = RandomForestClassifier(oob_score=metric, random_state=0)
    clf.fit(X, y)
    desired_oob_score = clf.oob_score_
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)
    test_oob_score = rfc_utils.oob_score(clf, X, y, metric, n_jobs=2)
    assert_allclose(test_oob_score, desired_oob_score)


@pytest.mark.parametrize(
    "class_weights, expected",
    [
        (None, 0.82),
        ([0.2, 0.8], 0.75),
    ],
)
def test_oob_score_proba(class_weights, expected):
    X, y = make_classification(
        n_samples=10, n_features=10, weights=class_weights, random_state=0
    )
    X = pd.DataFrame(X)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)
    test_oob_score = rfc_utils.oob_score(
        clf, X, y, roc_auc_score, n_jobs=2, scoring_on_pred=False
    )
    assert_allclose(test_oob_score, expected)


def test_oob_score_require_bootstrap():
    X, y = make_classification(n_samples=10, n_features=10, random_state=0)
    X = pd.DataFrame(X)
    clf = RandomForestClassifier(random_state=0, bootstrap=False)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="if bootstrap"):
        rfc_utils.oob_score(clf, X, y, roc_auc_score, n_jobs=2, scoring_on_pred=False)


@pytest.mark.parametrize(
    "name", ["max_depth", "criterion", "class_weight", "fake_parameter"]
)
@given(val=st.floats(allow_nan=False, allow_infinity=False))
@example(val=-1.0)
@example(val=12.0)
def test_floatparams(val, name):
    if val < 0.0:
        with pytest.raises(ValueError, match="value < 0"):
            rfc_utils._floatparam(name, val)
    elif name in ["criterion", "class_weight"] and val > 1.0:
        with pytest.raises(ValueError, match="in range"):
            rfc_utils._floatparam(name, val)
    elif name == "fake_parameter":
        with pytest.raises(ValueError, match="parameter not recognized"):
            rfc_utils._floatparam(name, val)
    else:
        RandomForestClassifier(**{name: rfc_utils._floatparam(name, val)})


@pytest.mark.parametrize("random_state, score", [(0, 1.0), (1, 1.0), (2, 0.25)])
def test_min_cv_score(random_state, score):
    X, y = make_classification(n_samples=10, n_features=10, random_state=random_state)
    test_score = rfc_utils.min_cv_score(
        X, y, cv=3, n_estimators=10, random_state=random_state
    )
    assert_allclose(test_score, score)


@given(
    max_depth=st.floats(
        min_value=0.0, max_value=100.99999, allow_nan=False, allow_infinity=False
    ),
    criterion=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    class_weight=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    floats=st.booleans(),
)
def test_min_cv_score_args(max_depth, criterion, class_weight, floats):
    X, y = make_classification(n_samples=10, n_features=10)
    if not floats:
        max_depth = rfc_utils._floatparam("max_depth", max_depth)
        criterion = rfc_utils._floatparam("criterion", criterion)
        class_weight = rfc_utils._floatparam("class_weight", class_weight)
    rfc_utils.min_cv_score(
        X,
        y,
        cv=3,
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
def test_min_cv_score_args_passed(max_depth, criterion, class_weight):
    X, y = make_classification(n_samples=10, n_features=10)
    with pytest.raises(ValueError, match="InvalidParameterError"):
        rfc_utils.min_cv_score(
            X,
            y,
            cv=3,
            max_depth=max_depth,
            criterion=criterion,
            class_weight=class_weight,
            n_estimators=10,
        )


@pytest.mark.parametrize(
    "distributions, score, params",
    [
        (
            None,
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
    res = rfc_utils.get_hyperparameters(
        X,
        y,
        init_points=1,
        n_iter=1,
        n_jobs=1,
        random_state=0,
        distributions=distributions,
        n_estimators=10,
    )
    assert res["target"] == pytest.approx(score)
    assert_array_equal(
        np.sort(list(res["params"].keys())), np.sort(list(params.keys()))
    )
    for key in res["params"].keys():
        assert res["params"][key] == pytest.approx(params[key])
