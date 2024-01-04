import numpy as np
import pandas as pd
import sklearn.ensemble._forest as forest_utils
from joblib import Parallel, delayed


def calc_pred(est, X, n_samples, n_samples_bootstrap):
    # generate oob samples from stored random_state
    ind = forest_utils._generate_unsampled_indices(
        est.random_state, n_samples, n_samples_bootstrap
    )
    if isinstance(X, pd.DataFrame):
        y_pred = est.predict_proba(X.iloc[ind].to_numpy())
    else:
        y_pred = est.predict_proba(X[ind])
    return ind, y_pred


def oob_score(rfc, X, y, scorer, n_jobs=-1, scoring_on_pred=True):
    """Calculation of oob_score on a fit RandomForestClassifier utilizing parallelization
    Related to upstream issue https://github.com/scikit-learn/scikit-learn/issues/28059
    """
    n_samples = len(y)
    n_samples_bootstrap = forest_utils._get_n_samples_bootstrap(
        n_samples, rfc.max_samples
    )
    oob_pred = np.zeros(shape=(n_samples, rfc.n_classes_), dtype=np.float64)
    r = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(calc_pred)(est, X, n_samples, n_samples_bootstrap)
        for est in rfc.estimators_
    )
    for res in r:
        ind, y_pred = res
        oob_pred[ind, ...] += y_pred
    if scoring_on_pred:
        score = scorer(y, np.argmax(oob_pred, axis=1))
    else:
        for proba in oob_pred:
            proba /= len(rfc.estimators_)
        # only written with binary classification in mind
        y_scores = oob_pred[:, 1].reshape(-1, 1)
        score = scorer(y, y_scores)
    return score
