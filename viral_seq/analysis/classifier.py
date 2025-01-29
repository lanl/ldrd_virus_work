import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Union, Any, Optional, NamedTuple
from viral_seq.analysis import spillover_predict as sp
from joblib import Parallel, delayed
from sklearn.metrics import (
    get_scorer,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler
import pickle
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from shapely.geometry import LineString


class CV_data(NamedTuple):
    y_tests: list[np.ndarray]
    y_preds: list[np.ndarray]
    scores: npt.NDArray[np.floating[Any]]


class CV_ROC_data(NamedTuple):
    mean_tpr: np.ndarray
    tpr_std: np.ndarray
    tpr_folds: list[np.ndarray]
    fpr_folds: Optional[list[np.ndarray]]
    tpr_std_folds: Optional[list[np.ndarray]]


class EER_Data(NamedTuple):
    eer_threshold_index: np.int32
    eer_threshold: np.float64
    eer_value: np.float64


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
        "group": "RandomForestClassifier",
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
        "group": "LGBMClassifier_Boost",
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
    model_arguments["XGBClassifer Boost Seed:" + str(random_state)] = {
        "model": XGBClassifier,
        "suffix": "xgb_" + str(random_state),
        "group": "XGBClassifer_Boost",
        "optimize": {
            "num_samples": 350,
            "n_jobs_cv": 1,
            "config": {
                "n_jobs": 1,  # it's better to let ray handle parallelization
                "reg_alpha": tune.loguniform(0.001, 1000),
                "learning_rate": tune.uniform(0.1, 0.5),
                "min_child_weight": tune.loguniform(0.001, 120.0),
                "n_estimators": tune.randint(1, 4000),
                "subsample": tune.uniform(0.5, 1.0),
            },  # tunable ranges from https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html#xgboost-tunable-hyperparameters
        },
        "predict": {
            "n_jobs": n_jobs,
            "random_state": random_state,
        },
    }
    model_arguments["ExtraTreesClassifier Seed:" + str(random_state)] = {
        "model": ExtraTreesClassifier,
        "group": "ExtraTreesClassifier",
        "suffix": "etc_" + str(random_state),
        "optimize": {
            "num_samples": 1_500,
            "n_jobs_cv": 1,
            "config": {
                "bootstrap": True,
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
    model_arguments["LGBMClassifer Dart Seed:" + str(random_state)] = {
        "model": LGBMClassifier,
        "suffix": "lgbm_dart_" + str(random_state),
        "group": "LGBMClassifier_Dart",
        "optimize": {
            "num_samples": 2_000,
            "n_jobs_cv": 1,
            "config": {
                "verbose": -1,
                "force_col_wise": True,
                "n_estimators": 500,
                "n_jobs": 1,  # it's better to let ray handle parallelization
                "boosting_type": "dart",
                "num_leaves": tune.randint(10, 100),
                "learning_rate": tune.loguniform(1e-3, 0.01),
                "subsample": tune.uniform(0.1, 1.0),
                "subsample_freq": tune.randint(0, 10),
                "max_depth": tune.randint(15, 100),
                "min_child_samples": tune.randint(10, 200),
                "colsample_bytree": tune.uniform(0.1, 1.0),
                "drop_rate": tune.uniform(0.0, 1.0),
                "skip_drop": tune.uniform(0.0, 1.0),
            },  # Boost space with Dart specific drop_rate, skip_drop
        },
        "predict": {
            "boosting_type": "dart",
            "verbose": -1,
            "force_col_wise": True,
            "n_jobs": n_jobs,
            "random_state": random_state,
        },
    }
    return model_arguments


def _cv_child(
    model, X, y, scoring, train, test, **kwargs
) -> tuple[pd.Series, npt.ArrayLike, float]:
    X_train = X.iloc[train]
    # After splitting data, kmers may no longer be shared in training
    X_train = sp.drop_unshared_kmers(X_train)
    X_test = X.iloc[test]
    X_test = X_test[X_train.columns]
    clf = model(**kwargs)
    clf.fit(X_train, y[train])
    scorer = get_scorer(scoring)
    y_pred = clf.predict_proba(X_test)[..., 1]
    score = scorer(clf, X_test, y[test])
    return y[test], y_pred, score


def cross_validation(
    model,
    X: pd.DataFrame,
    y: npt.ArrayLike,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs_cv: int = 1,
    **kwargs,  # classifier arguments
) -> CV_data:
    """Perform k-fold cross-validation stratified on target

    Parameters:
        model: model to use for training, expects sklearn-like interface
        X (pd.DataFrame), y (npt.ArrayLike): data used for model evaluation
        n_splits (int): number of folds to use
        scoring (str): scoring method accepted by `sklearn.metrics.get_scorer`
        n_jobs_cv (int): number of jobs to run in parallel
        **kwargs: passed to model

    Returns:
        y_tests (list[np.ndarray]): each test set for each fold
        y_preds (list[np.ndarray]): corresponding predictions for each test set
        scores (np.ndarray(np.floating[Any]): score achieved for each fold.
    """
    cv = StratifiedKFold(n_splits=n_splits)
    scores = np.zeros(n_splits)
    y_preds = []
    y_tests = []
    r = Parallel(n_jobs=n_jobs_cv)(
        delayed(_cv_child)(model, X, y, scoring, train, test, **kwargs)
        for train, test in cv.split(X, y)
    )
    for i, res in enumerate(r):
        y_tests.append(res[0])
        y_preds.append(res[1])
        scores[i] = res[2]
    return CV_data(y_tests, y_preds, scores)


def cv_score(
    model,
    X: pd.DataFrame,
    y: npt.ArrayLike,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs_cv: int = 1,
    **kwargs,  # classifier arguments
) -> float:
    cv_data = cross_validation(model, X, y, n_splits, scoring, n_jobs_cv, **kwargs)
    return np.mean(cv_data.scores)


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
    y_test,
    name="Classifier",
    model_out="",
    params_predict=None,
    params_optimized=None,
    calibrate=True,
    filename_calibration_curve="plot_calibration_curve.png",
):
    """Trains model with X_train, y_train, then predicts on X_test. We expect params_optimized come from an optimization procedure which may have parameters we want to adjust using params_predict.

    Parameters:
        model: expects sklearn-like interface
        X_train: any form model accepts for fitting
        y_train: any form model accepts for fitting
        X_test: any form model accepts for prediction
        y_test: any form accepted by `sklearn.calibration.CalibrationDisplay.from_predictions`
        name: customizes print statement for clarity with multiple calls
        model_out: if not empty, fitted model is pickled to this filename
        params_predict: model parameters to use which supersede params_optimized
        params_optimized: model parameters to use if not overriden by params_predict
        calibrate: whether or not to check calibration with `sklearn.calibration.CalibratedClassifierCV`
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
    y_pred = clf.predict_proba(X_test)[..., 1]
    this_auc = roc_auc_score(y_test, y_pred)
    print(f"{name} achieved ROC AUC = {this_auc:.2f} on test data.")
    y_pred_calibrated = None
    if calibrate:
        print("Calibrating model with cross-validation")
        clf_unfit = model(**params_predict, **params_optimized)
        clf_calibrated = CalibratedClassifierCV(clf_unfit)
        clf_calibrated.fit(X_train, y_train)
        y_pred_calibrated = clf_calibrated.predict_proba(X_test)[..., 1]
        this_auc_calibrated = roc_auc_score(y_test, y_pred_calibrated)
        print(
            f"{name} achieved ROC AUC = {this_auc_calibrated:.2f} on test data after calibration with cross-validation."
        )
    plot_calibration_curve(
        y_test,
        y_pred,
        y_pred_calibrated,
        title=f"Calibration Curve\n{name}",
        filename=filename_calibration_curve,
    )
    if calibrate:
        y_pred = y_pred_calibrated
        clf = clf_calibrated
        # we expected feature_importances_ here later
        clf.feature_importances_ = np.mean(
            [
                clf.calibrated_classifiers_[i].estimator.feature_importances_
                for i in range(len(clf.calibrated_classifiers_))
            ],
            axis=0,
        )
    if model_out:
        with open(model_out, "wb") as f:
            print("Saving trained model to", model_out)
            pickle.dump(clf, f)
    return clf, y_pred


def get_roc_curve(
    y_true: npt.ArrayLike, predictions: dict[str, npt.ArrayLike]
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """If passed one set of predictions, this functions identically to `sklearn.metrics.roc_curve`. If passed multiple sets of predictions, this instead returns a mean roc curve with standard deviation."""
    fprs, tprs, tpr_std = [], [], None
    for key in predictions:
        fpr, tpr, thresh = roc_curve(y_true, predictions[key])
        tprs.append(tpr)
        fprs.append(fpr)
    if len(tprs) > 1:
        mean_fpr = np.linspace(0, 1, 100)
        for i in range(len(tprs)):
            tprs[i] = np.interp(mean_fpr, fprs[i], tprs[i])
            tprs[i][0] = 0.0
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        tpr_std = np.std(tprs, axis=0)
        tpr = mean_tpr
        fpr = mean_fpr
    return fpr, tpr, tpr_std


def get_roc_curve_cv(
    y_trues: list[npt.ArrayLike], predictions: list[list[np.ndarray]]
) -> CV_ROC_data:
    """Like `sklearn.metrics.roc_curve` designed to handle multiple copies of k-fold cross-validation data.

    Parameters:
        y_trues (list[npt.ArrayLike]): Truth values of the cross-validation test sets. len(y_trues) is number of folds, with each y_true[i] corresponding to the y_true of the ith fold
        predictions (list[list[npt.ArrayLike]]): Cross-validation predictions. len(predictions) is the number of copies. len(predictions[i]) is the number of folds.

    Returns:
        mean_tpr (np.ndarray): Average tpr across all folds, copies
        tpr_std (np.ndarray): Standard deviation of mean_tpr
        tpr_folds (list[np.ndarray]): len(mean_tpr_folds) is number of folds. If copies > 1, average tpr of each fold, else tpr of each fold
        fpr_folds (Optional[list[np.ndarray]]): If copies > 1 this is None, else this is the fpr of each fold
        tpr_std_folds (Optional[list[np.ndarray]]): If copies > 1, standard deviation of each tpr_folds, else None

    """

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    n_folds = len(predictions[0])
    average_folds = len(predictions) > 1
    fpr_folds: list[list] = [[] for _ in range(n_folds)]
    tpr_folds: list[list] = [[] for _ in range(n_folds)]
    ret_fpr_folds: Optional[list[np.ndarray]] = None if average_folds else []
    ret_tpr_folds = []
    tpr_std_folds: Optional[list[np.ndarray]] = [] if average_folds else None
    for folds in predictions:
        for i, this_pred in enumerate(folds):
            # get each roc_curve
            this_fpr, this_tpr, thresh = roc_curve(y_trues[i], this_pred)
            interp_tpr = np.interp(mean_fpr, this_fpr, this_tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            tpr_folds[i].append(this_tpr)
            fpr_folds[i].append(this_fpr)
    # average per fold
    for i in range(n_folds):
        if average_folds:
            for j in range(len(tpr_folds[i])):
                tpr_folds[i][j] = np.interp(mean_fpr, fpr_folds[i][j], tpr_folds[i][j])
                tpr_folds[i][j][0] = 0.0
            ret_tpr_folds.append(np.mean(tpr_folds[i], axis=0))
            ret_tpr_folds[-1][-1] = 1.0
            tpr_std_folds.append(np.std(tpr_folds[i], axis=0))  # type: ignore
        else:
            ret_tpr_folds.append(tpr_folds[i][0])
            ret_fpr_folds.append(fpr_folds[i][0])  # type: ignore
    # average all copies & folds together
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    tpr_std = np.std(tprs, axis=0)
    return CV_ROC_data(
        mean_tpr=mean_tpr,
        tpr_std=tpr_std,
        tpr_folds=ret_tpr_folds,
        fpr_folds=ret_fpr_folds,
        tpr_std_folds=tpr_std_folds,
    )


def plot_roc_curve(
    name: str,
    fpr: np.ndarray,
    tpr: np.ndarray,
    tpr_std: Optional[np.ndarray] = None,
    filename: str = "roc_plot.png",
    title: str = "ROC curve",
    eer_data: Optional[EER_Data] = None,
):
    tpr_stds = None if tpr_std is None else [tpr_std]
    eer_data_list = None if eer_data is None else [eer_data]
    plot_roc_curve_comparison(
        [name], [fpr], [tpr], tpr_stds, filename, title, eer_data_list
    )


def plot_roc_curve_comparison(
    names: list[str],
    fprs: Optional[list[np.ndarray]],
    tprs: list[np.ndarray],
    tpr_stds: Optional[list[np.ndarray]] = None,
    filename: str = "roc_plot_comparison.png",
    title: str = "ROC curve",
    eer_data_list: Optional[list[EER_Data]] = None,
):
    """Can plot one or multiple roc curves. Will plot a plus/minus 1 standard deviation shaded region for curves if provided."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if fprs is None:
        fprs = [np.linspace(0, 1, 100) for _ in range(len(tprs))]
    eer_line = False
    for i, name in enumerate(names):
        this_auc = auc(fprs[i], tprs[i])
        ax.plot(fprs[i], tprs[i], label=f"{name} (AUC = {this_auc:.2f})")
        if tpr_stds and tpr_stds[i] is not None:
            tprs_upper = np.minimum(tprs[i] + tpr_stds[i], 1)
            tprs_lower = np.maximum(tprs[i] - tpr_stds[i], 0)
            ax.fill_between(
                fprs[i],
                tprs_lower,
                tprs_upper,
                label=f"{name} \u00B11 std. dev.",
                alpha=0.2,
            )
        if eer_data_list and eer_data_list[i] is not None:
            eer_line = True
            eer_threshold = eer_data_list[i].eer_threshold
            eer_threshold_index = eer_data_list[i].eer_threshold_index
            eer_x = fprs[i][eer_threshold_index]
            eer_y = tprs[i][eer_threshold_index]
            # assume point is below eer line
            next_idx = eer_threshold_index + 1
            if eer_x + eer_y > 1.0:
                # above eer line case
                next_idx = eer_threshold_index - 1
            # find intercept between this line segment and eer line
            eer_next_x = fprs[i][next_idx]
            eer_next_y = tprs[i][next_idx]
            this_line = LineString([(eer_x, eer_y), (eer_next_x, eer_next_y)])
            intersection = this_line.intersection(LineString([(0, 1), (1, 0)]))
            ax.plot(
                intersection.x,  # type: ignore
                intersection.y,  # type: ignore
                marker="x",
                label=f"{name} {eer_threshold = :.2e}",
                alpha=1.0,
                ms=12,
            )

    # chance line
    ax.plot([0, 1], [0, 1], "r--")
    # EER line
    if eer_line:
        ax.plot([0, 1], [1, 0], "--", color="grey", alpha=0.5)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.axis("square")
    ax.legend(loc="lower right")

    fig.savefig(filename, dpi=300)
    plt.close()


def plot_calibration_curve(
    y_test: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    y_pred_calibrated: Optional[npt.ArrayLike] = None,
    title: str = "Calibration Curve",
    filename: str = "plot_calibration_curve.png",
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)
    ax.plot(prob_pred, prob_true, label="Predictions on test", marker="s")
    if y_pred_calibrated is not None:
        prob_true_calibrated, prob_pred_calibrated = calibration_curve(
            y_test, y_pred_calibrated, n_bins=10
        )
        ax.plot(
            prob_pred_calibrated,
            prob_true_calibrated,
            label="Predictions on test (calibrated with CV)",
            marker="s",
        )

    ax.set(
        ylabel="Fraction of positives (Positive class: 1)",
        xlabel="Mean predicted probability (Positive class: 1)",
        title=title,
    )

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close()


def cal_eer_thresh_and_val(
    fpr: np.ndarray, tpr: np.ndarray, threshold: np.ndarray
) -> EER_Data:
    # see EER details: https://stackoverflow.com/a/46026962/2942522
    # also Probabilistic Machine Learning: An Introduction by Kevin P. Murphy, section 5.7.2.1
    fnr = 1 - tpr
    threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer_threshold = threshold[threshold_index]
    eer_value = fpr[threshold_index]
    eer_data = EER_Data(
        eer_threshold_index=threshold_index,
        eer_threshold=eer_threshold,
        eer_value=eer_value,
    )
    return eer_data
