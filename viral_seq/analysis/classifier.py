import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Union, Any
from viral_seq.analysis import spillover_predict as sp
from joblib import Parallel, delayed
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler
import pickle
from lightgbm import LGBMClassifier


def get_model_arguments(
    n_jobs: int, random_state: int, num_samples: int, num_features: int
):
    """A simple helper function to store model parameters"""
    one_sample = 1.0 / num_samples
    one_feature = 1.0 / num_features
    sqrt_feature = (num_features) ** 0.5 / num_features
    model_arguments = {}
    model_arguments["RandomForestClassifier Seed:" + str(random_state)] = {
        "model": RandomForestClassifier,
        "suffix": "rfc_" + str(random_state),
        "optimize": {
            "num_samples": 3_000,
            "n_jobs_cv": 1,
            "config": {
                "n_estimators": 2_000,
                "n_jobs": 1,  # it's better to let ray handle parallelization
                "max_samples": tune.uniform(one_sample, 1.0),
                "min_samples_leaf": tune.uniform(
                    one_sample, np.min([1.0, 10 * one_sample])
                ),
                "min_samples_split": tune.uniform(
                    one_sample, np.min([1.0, 300 * one_sample])
                ),
                "max_features": tune.uniform(
                    one_feature, np.min([1.0, 2 * sqrt_feature])
                ),
                "criterion": tune.choice(
                    ["gini", "log_loss"]
                ),  # no entropy, see Geron ISBN 1098125975 Chapter 6
                "class_weight": tune.choice([None, "balanced", "balanced_subsample"]),
                "max_depth": tune.choice([None] + [i for i in range(1, 31)]),
            },
        },
        "predict": {
            "n_estimators": 10_000,
            "n_jobs": n_jobs,
            "random_state": random_state,
        },
    }
    model_arguments["LGBMClassifer Boost Seed:" + str(random_state)] = {
        "model": LGBMClassifier,
        "suffix": "lgbm_" + str(random_state),
        "optimize": {
            "num_samples": 2_000,
            "n_jobs_cv": 1,
            "config": {
                "verbose": -1,
                "force_col_wise": True,
                "n_estimators": 500,
                "n_jobs": 1,  # it's better to let ray handle parallelization
                "num_leaves": tune.randint(10, 100),
                "learning_rate": tune.loguniform(1e-3, 0.01),
                "subsample": tune.uniform(0.1, 1.0),
                "subsample_freq": tune.randint(0, 10),
                "max_depth": tune.randint(15, 100),
                "min_child_samples": tune.randint(10, 200),
                "colsample_bytree": tune.uniform(0.1, 1.0),
            },  # tunable ranges from https://docs.aws.amazon.com/sagemaker/latest/dg/lightgbm-tuning.html#lightgbm-tunable-hyperparameters
        },
        "predict": {
            "verbose": 1,
            "n_jobs": n_jobs,
            "random_state": random_state,
        },
    }
    return model_arguments


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


def train_and_predict(
    model,
    X_train,
    y_train,
    X_test,
    name="Classifier",
    model_out="",
    params_predict=None,
    params_optimized=None,
):
    """Trains model with X_train, y_train, then predicts on X_test. We expect params_optimized come from an optimization procedure which may have parameters we want to adjust using params_predict.

    Parameters:
        model: expects sklearn-like interface
        X_train: any form model accepts for fitting
        y_train: any form model accepts for fitting
        X_test: any form model accepts for prediction
        name: customizes print statement for clarity with multiple calls
        model_out: if not empty, fitted model is pickled to this filename
        params_predict: model parameters to use which supersede params_optimized
        params_optimized: model parameters to use if not overriden by params_predict
    Returns:
        fitted model and predictions on X_test
    """
    if params_predict is None:
        params_predict = {}
    if params_optimized is None:
        params_optimized = {}
    params_optimized = {
        k: v for k, v in params_optimized.items() if k not in params_predict
    }
    print(
        "Will train model and run prediction on test for",
        name,
        "using the following parameters:",
    )
    print({**params_predict, **params_optimized})
    clf = model(**params_predict, **params_optimized)
    clf.fit(X_train, y_train)
    if model_out:
        with open(model_out, "wb") as f:
            print("Saving trained model to", model_out)
            pickle.dump(clf, f)
    y_pred = clf.predict_proba(X_test)[..., 1]
    return clf, y_pred
