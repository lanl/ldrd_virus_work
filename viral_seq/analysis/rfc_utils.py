import numpy as np
import pandas as pd
import sklearn.ensemble._forest as forest_utils
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import math


_model = RandomForestClassifier
_floatparam_names = ["class_weight", "criterion", "max_depth"]


def _default_parameters(shape: tuple[int, int]):
    """Returns a dictionary of default values in float form to use for hyperparameter optimization.
    Not all-inclusive; only parameters we intend to do hyperparameter optimization on are listed.
    """
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape dimensions must be positive, instead got", shape)
    one_sample = 1.0 / shape[0]
    sqrt_feature = np.sqrt(shape[1]) / shape[1]
    return {
        "criterion": 0.0,  # gini
        "max_depth": 0.0,  # None
        "min_samples_split": 2.0 * one_sample,
        "min_samples_leaf": one_sample,
        "max_features": sqrt_feature,
        "max_samples": 1.0,
        "class_weight": 0.0,  # None
    }


def calc_pred(est, X: pd.DataFrame, n_samples, n_samples_bootstrap):
    # generate oob samples from stored random_state
    ind = forest_utils._generate_unsampled_indices(
        est.random_state, n_samples, n_samples_bootstrap
    )
    y_pred = est.predict_proba(X.iloc[ind].to_numpy())
    return ind, y_pred


def oob_score(rfc, X: pd.DataFrame, y, scorer, n_jobs=-1, scoring_on_pred=True):
    """Calculation of oob_score on a fit RandomForestClassifier utilizing parallelization
    Related to upstream issue https://github.com/scikit-learn/scikit-learn/issues/28059
    """
    if not rfc.get_params()["bootstrap"]:
        raise ValueError("Out of bag estimation only available if bootstrap=True")
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


def _floatparam(name: str, val: float):
    """Utility function to allow bayesian optimization of RandomForestClassifer parameters that don't accept float values"""
    if val < 0.0:
        raise ValueError("No RandomForestClassifier parameter accept a value < 0")
    # finite parameters
    options = []
    if name == "class_weight":
        options = [None, "balanced", "balanced_subsample"]
    elif name == "criterion":
        options = ["gini", "entropy", "log_loss"]
    this_len = len(options)
    if this_len > 0:
        if val > 1.0:
            raise ValueError(
                "If parameter has a finite number of output values, float value should be in range [0,1]"
            )
        val = float(
            math.floor(np.interp(val, [0, 1], [0, this_len]))
        )  # float() to satisfy mypy
        val = (
            this_len - 1 if val == this_len else val
        )  # bayes_opt uses closed bounds only
        return options[int(val)]

    if name == "max_depth":
        return int(math.floor(val)) if val >= 1.0 else None

    raise ValueError(
        "RandomForestClassifier parameter not recognized. This functionality may not be implemented for this parameter. If this parameter accepts a float by default this function shouldn't be used."
    )
