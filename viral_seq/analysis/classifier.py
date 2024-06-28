import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Union, Any
from viral_seq.analysis import spillover_predict as sp
from joblib import Parallel, delayed
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler


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
    model,
    X,
    y,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs_cv: int = 1,
    **kwargs,  # classifier arguments
) -> float:
    cv = StratifiedKFold(n_splits=n_splits)
    scores = []
    r = Parallel(n_jobs=n_jobs_cv)(
        delayed(_cv_score_child)(model, X, y, scoring, train, test, **kwargs)
        for train, test in cv.split(X, y)
    )
    for res in r:
        scores.append(res)
    return np.mean(scores)


def _tune_objective(config, **kwargs):
    score = cv_score(**config, **kwargs)
    ray.train.report({"mean_roc_auc": score})


def get_hyperparameters(
    model,
    X: pd.DataFrame,
    y: npt.ArrayLike,
    config: dict[str, tuple[float, float]],
    num_samples: int = 1,
    random_state: int = 0,
    n_jobs_cv: int = 1,
    max_concurrent_trials: int = 0,
) -> dict[str, Union[float, dict[str, Any]]]:
    """Search for the best hyperparameters using `ray.tune`

    Parameters:
        model: model to use for training, expects sklearn-like interface
        X (pd.DataFrame), y (npt.ArrayLike): data used for model evaluation
        config (dict[str, tuple[float, float]]): dictionary of parameters to optimize
        num_samples (int): number of hyperparameters to sample from parameter space
        random_state (int): seed used when needed for repeatability
        n_jobs_cv (int): used for `sklearn.model_selection.cross_val_score`, pass n_jobs to model training with `model_parameters`
        max_concurrent_trials (int): limits concurrency, this is mostly used for testing as `optuna` is only repeatable when serial

    Returns:
        (dict[str, Union[float, dict[str, Any]]]): dictionary with best performing score (key 'target'), parameters (key 'params'), and target history during optimization (key 'targets')
    """
    algo = OptunaSearch(
        metric="mean_roc_auc", mode="max", sampler=TPESampler(seed=random_state)
    )
    put_X = ray.put(X)
    put_y = ray.put(y)
    results = tune.run(
        tune.with_parameters(
            _tune_objective,
            model=model,
            X=put_X,
            y=put_y,
            n_jobs_cv=n_jobs_cv,
            random_state=random_state,
        ),
        search_alg=algo,
        metric="mean_roc_auc",
        mode="max",
        num_samples=num_samples,
        config=config,
        verbose=1,
        max_concurrent_trials=max_concurrent_trials,
    )
    res = {}
    best_result = results.best_result
    res_df = results.dataframe()
    res["params"] = results.best_config
    res["target"] = best_result["mean_roc_auc"]
    targets = list(res_df["mean_roc_auc"].values)
    res["targets"] = targets
    return res
