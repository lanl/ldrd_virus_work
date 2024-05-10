import numpy as np
import numpy.typing as npt
import pandas as pd
from functools import partial
from bayes_opt import BayesianOptimization
from typing import Union, Any
from viral_seq.analysis import spillover_predict as sp
from joblib import Parallel, delayed
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold


def _cv_score_child(model, X, y, scoring, train, test, **kwargs) -> float:
    X_train = X.iloc[train]
    # After splitting data, kmers may no longer be shared in training
    X_train = sp.drop_unshared_kmers(X_train)
    X_test = X.iloc[test]
    X_test = X_test[X_train.columns]
    clf = model(**kwargs)
    clf.fit(X_train, y[train])
    scorer = get_scorer(scoring)
    return scorer(clf, X_test, y[test])


def cv_score(
    model_utils,
    X,
    y,
    n_splits=5,
    scoring: str = "roc_auc",
    n_jobs_cv: int = 1,
    **kwargs,  # classifier arguments
) -> float:
    for param in model_utils._floatparam_names:
        if param in kwargs and isinstance(kwargs[param], float):
            kwargs[param] = model_utils._floatparam(param, kwargs[param])
    cv = StratifiedKFold(n_splits=n_splits)
    scores = []
    r = Parallel(n_jobs=n_jobs_cv)(
        delayed(_cv_score_child)(
            model_utils._model, X, y, scoring, train, test, **kwargs
        )
        for train, test in cv.split(X, y)
    )
    for res in r:
        scores.append(res)
    return np.mean(scores)


def get_hyperparameters(
    model_utils,
    X: pd.DataFrame,
    y: npt.ArrayLike,
    distributions: dict[str, tuple[float, float]],
    model_parameters: dict[str, Any] = {},
    bayes_parameters: dict[str, Any] = {},
    random_state: int = 0,
    n_jobs_cv: int = 1,
) -> dict[str, Union[float, dict[str, Any]]]:
    """Search for the best hyperparameters using bayesian optimization

    Parameters:
        model_utils: helper module for each model, see use below and `viral_seq.analysis.rfc_utils`
        X (pd.DataFrame), y (npt.ArrayLike): data used for model evaluation
        distributions (dict[str, tuple[float, float]]): dictionary of parameters to optimize
        model_parameters: passed as is to the model used for scoring. These are not optimized and should not overlap with parameters in `distributions`
        bayes_parameters: passed as is to `bayes_opt.BayesianOptimization.maximize`
        random_state (int): seed used when needed for repeatability
        n_jobs_cv (int): used for `sklearn.model_selection.cross_val_score`, pass n_jobs to model training with `model_parameters`

    Returns:
        (dict[str, Union[float, dict[str, Any]]]): dictionary with best performing score (key 'target'), parameters (key 'params'), and target history during optimization (key 'targets')
    """
    duplicate_params = set(distributions.keys()).intersection(set(model_parameters))
    if len(duplicate_params) > 0:
        raise ValueError(
            "Received inputs for the following parameters in both `distributions` and `model_parameters`: "
            + str(duplicate_params)
            + ". Model parameters to be optimized should be passed in `distributions` and those to be set but not optimized in `model_parameters`."
        )
    optimizer = BayesianOptimization(
        partial(
            cv_score,
            model_utils,
            X=X,
            y=y,
            n_jobs_cv=n_jobs_cv,
            random_state=random_state,
            **model_parameters,
        ),
        distributions,
        random_state=random_state,
    )
    # check default settings first
    defaults = model_utils._default_parameters(X.shape)
    # only include what's in the parameter space
    defaults = {key: defaults[key] for key in distributions.keys()}
    optimizer.probe(params=defaults, lazy=True)
    optimizer.maximize(**bayes_parameters)
    res = optimizer.max
    for name in model_utils._floatparam_names:
        if name in res["params"]:
            res["params"][name] = model_utils._floatparam(name, res["params"][name])
    targets = list(pd.DataFrame(optimizer.res)["target"].values)
    res["targets"] = targets
    return res
