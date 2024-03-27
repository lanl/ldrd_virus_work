import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.ensemble._forest as forest_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
from functools import partial
from bayes_opt import BayesianOptimization
from typing import Optional, Union, Any
import math


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


def min_cv_score(
    X: npt.ArrayLike, y: npt.ArrayLike, cv: int = 5, scoring: str = "roc_auc", **kwargs
) -> float:
    """Perform cv-fold cross validation and return the minimum of the scores across folds"""
    for name in ["max_depth", "criterion", "class_weight"]:
        if name in kwargs and isinstance(kwargs[name], float):
            kwargs[name] = _floatparam(name, kwargs[name])
    return cross_val_score(
        RandomForestClassifier(**kwargs), X, y, scoring=scoring, cv=cv
    ).min()


def get_hyperparameters(
    X: npt.NDArray,
    y: npt.NDArray,
    init_points: int = 10,
    n_iter: int = 20,
    n_jobs: int = 1,
    random_state: int = 0,
    distributions: Optional[dict[str, tuple[float, float]]] = None,
    **kwargs
) -> dict[str, Union[float, dict[str, Any]]]:
    """Search for the best hyperparameters using bayesian optimization

    Parameters:
        X (npt.NDArray), y (npt.NDArray): passed as is to `sklearn.model_selection.cross_val_score`
        init_points (int), n_iter (int): passed as is to `bayes_opt.BayesianOptimization.maximize`
        random_state (int): seed used when needed for repeatability
        distributions (Optional[dict[str, tuple[float, float]]]): dictionary of parameters to optimize
        n_jobs (int), **kwargs: passed to the `sklearn.ensemble.RandomForestClassifer` used for scoring

    Returns:
        (dict[str, Union[float, dict[str, Any]]]): dictionary with best performing score (key 'target') and parameters (key 'params')
    """
    one_sample = 1.0 / X.shape[0]
    sqrt_feature = np.sqrt(X.shape[1]) / X.shape[1]
    one_feature = 1.0 / X.shape[1]
    if distributions is None:
        distributions = {
            "max_samples": (one_sample, 1.0),
            "min_samples_leaf": (one_sample, np.min([1.0, 10 * one_sample])),
            "min_samples_split": (one_sample, np.min([1.0, 100 * one_sample])),
            "max_features": (one_feature, np.min([1.0, 2 * sqrt_feature])),
            "criterion": (0.0, 1.0),
            "class_weight": (0.0, 1.0),
            "max_depth": (0.0, 30.99999),  # <1.0 is None
        }
    optimizer = BayesianOptimization(
        partial(
            min_cv_score,
            X=X,
            y=y,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        ),
        distributions,
        random_state=random_state,
    )
    # check default settings first
    defaults = {
        "max_samples": 1.0,
        "min_samples_leaf": one_sample,
        "min_samples_split": 2.0 * one_sample,
        "max_features": sqrt_feature,
        "criterion": 0.0,  # gini
        "class_weight": 0.0,  # None
        "max_depth": 0.0,  # None
    }
    # only include what's in the parameter space
    defaults = {key: defaults[key] for key in distributions.keys()}
    optimizer.probe(params=defaults, lazy=True)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    res = optimizer.max
    for name in ["max_depth", "criterion", "class_weight"]:
        if name in res["params"]:
            res["params"][name] = _floatparam(name, res["params"][name])
    targets = list(pd.DataFrame(optimizer.res)["target"].values)
    res["targets"] = targets
    return res
