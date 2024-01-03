from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from viral_seq.analysis import rfc_utils
from sklearn.metrics import accuracy_score


def test_oob_score():
    X, y = make_classification(n_samples=10, n_features=10, random_state=0)
    clf = RandomForestClassifier(oob_score=True)
    clf.fit(X, y)
    test_oob_score = rfc_utils.oob_score(clf, X, y, accuracy_score)
    assert clf.oob_score_ == test_oob_score
