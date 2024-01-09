from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from viral_seq.analysis import rfc_utils
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from numpy.testing import assert_allclose
import pandas as pd
import pytest


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
