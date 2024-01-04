from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from viral_seq.analysis import rfc_utils
from sklearn.metrics import accuracy_score, roc_auc_score
from numpy.testing import assert_allclose
import pandas as pd


def test_oob_score():
    X, y = make_classification(n_samples=10, n_features=10, random_state=0)
    X = pd.DataFrame(X)
    # we can check accuracy_score against built-in sklearn oob_score
    clf = RandomForestClassifier(oob_score=True, random_state=0)
    clf.fit(X, y)
    desired_oob_score = clf.oob_score_
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)
    test_oob_score = rfc_utils.oob_score(clf, X, y, accuracy_score, n_jobs=2)
    assert_allclose(test_oob_score, desired_oob_score)
    # can only check scoring on y_scores with hardcoded regression test
    test_oob_score = rfc_utils.oob_score(
        clf, X, y, roc_auc_score, n_jobs=2, scoring_on_pred=False
    )
    assert_allclose(test_oob_score, 0.82)
