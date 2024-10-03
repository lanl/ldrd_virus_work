from collections import defaultdict
from typing import Optional, Sequence, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

matplotlib.use("Agg")


def sort_features(
    feature_importances: npt.NDArray[np.float64],
    feature_names: npt.NDArray,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray]:
    sort_idx = np.argsort(feature_importances)
    return feature_importances[sort_idx], feature_names[sort_idx]


def feature_importance_consensus(
    pos_class_feat_imps: Sequence[npt.NDArray[np.float64]],
    feature_names: npt.NDArray,
    top_feat_count: int,
) -> Tuple[npt.NDArray, npt.NDArray[np.int64], int]:
    """
    Parameters
    ----------
    pos_class_feat_imps: a sequence of NumPy arrays; each NumPy array corresponds
                         to either a shape (n_records, n_features) collection of SHAP values
                         for a given ML model (values are for the positive class
                         selection), or to a reduced version of this data structure
                         like with random forest feature importances with shape
                         (n_features,)
    feature_names: an array-like of strings of the features names of size ``n_features``
    top_feat_count: an integer representing the number of top features
                    to consider from each model when assessing the consensus

    Returns
    -------
    ranked_feature_names: array of feature names in descending order
                          of consensus importance (count) in the top
                          features per model
    ranked_feature_counts: array of counts (consensus occurrences) for
                           each feature in ``ranked_feature_names``
    num_input_models: int
    """
    num_input_models = len(pos_class_feat_imps)
    # calculate the mean absolute SHAP
    # values for each input ML model
    # OR simply the absolute values for already-reduced
    # feature importances
    processed_feat_imps = []
    for pos_class_imp_arr in pos_class_feat_imps:
        # for `shap`, handle new "Explanation" API https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/migrating-to-new-api.html#Migrating-to-the-new-%22Explanation%22-API
        if (
            hasattr(pos_class_imp_arr, "values")
            and np.atleast_2d(pos_class_imp_arr.values).shape[0] > 1
        ):
            # haven't reduced across the records yet (like raw SHAP importances)
            processed_feat_imps.append(
                np.mean(np.absolute(pos_class_imp_arr.values), axis=0)
            )
        elif np.atleast_2d(pos_class_imp_arr).shape[0] > 1:
            processed_feat_imps.append(np.mean(np.absolute(pos_class_imp_arr), axis=0))
        else:
            processed_feat_imps.append(np.absolute(pos_class_imp_arr))
    # for each input ML model store the
    # top_feat_count feature names
    top_feat_data: dict[str, int] = defaultdict(int)
    for processed_feat_imp in processed_feat_imps:
        sort_idx = np.argsort(processed_feat_imp)[::-1]
        top_feature_names = feature_names[sort_idx][:top_feat_count]
        for top_feature_name in top_feature_names:
            top_feat_data[top_feature_name] += 1
    top_feat_data = dict(
        sorted(top_feat_data.items(), key=lambda item: item[1], reverse=False)
    )
    ranked_feature_names = np.asarray(list(top_feat_data.keys()))
    ranked_feature_counts = np.asarray(list(top_feat_data.values()))
    return ranked_feature_names, ranked_feature_counts, num_input_models


def plot_feat_import(
    sorted_feature_importances: npt.NDArray[np.float64],
    sorted_feature_names: npt.NDArray,
    top_feat_count: int,
    model_name: str = "",
    fig_name_stem: str = "feat_imp",
):
    fig_name = fig_name_stem + ".png"
    fig_source = fig_name_stem + ".csv"
    df = pd.DataFrame()
    df["Feature Name"] = sorted_feature_names
    df["Feature Importance"] = sorted_feature_importances
    df.to_csv(fig_source, index=False)
    y_labels = sorted_feature_names[-top_feat_count:]
    x_data = sorted_feature_importances[-top_feat_count:]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    y_pos = np.arange(top_feat_count)
    ax.barh(y_pos, x_data)
    ax.set_xlabel("Feature Importance")
    ax.set_yticks(y_pos, labels=y_labels)
    ax.set_title(f"Feature importance for top {top_feat_count} features\n{model_name}")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300)  # type: ignore
    plt.close()


def plot_feat_import_consensus(
    ranked_feature_names: npt.NDArray,
    ranked_feature_counts: npt.NDArray[np.int64],
    num_input_models: int,
    top_feat_count: int,
    fig_name: Optional[str] = "feat_imp_consensus.png",
    fig_source: Optional[str] = "feat_imp_consensus.csv",
):
    df = pd.DataFrame()
    df["Feature Name"] = ranked_feature_names
    df[f"Count Of Models where ranked in top {top_feat_count} features"] = (
        ranked_feature_counts
    )
    df.to_csv(fig_source, index=False)
    y_labels = ranked_feature_names[-20:]
    x_vals = (ranked_feature_counts[-20:] / num_input_models) * 100
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    y_pos = np.arange(y_labels.size)
    ax.barh(y_pos, x_vals)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% ML models where ranked in top {top_feat_count} features")
    ax.set_yticks(y_pos, labels=y_labels)
    ax.set_title(f"Feature importance consensus amongst {num_input_models} models")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300)  # type: ignore
    plt.close()


def get_positive_shap_values(shap_values):
    # for the type handling here, see release 0.45.0 and
    # https://github.com/shap/shap/pull/3318
    if isinstance(shap_values, list):
        # legacy handling of `explainer.shap_values`
        positive_class_shap_values = shap_values[1]
    else:
        # shap 0.46.0 now returns np.ndarray for `explainer.shap_values`
        if (hasattr(shap_values, "values") and shap_values.values.ndim == 3) or (
            hasattr(shap_values, "ndim") and shap_values.ndim == 3
        ):
            positive_class_shap_values = shap_values[:, :, 1]
        else:
            # XGBoost case
            positive_class_shap_values = shap_values
    return positive_class_shap_values


def plot_shap_meanabs(
    shap_values,
    model_name: str = "",
    fig_name_stem: str = "feat_shap_meanabs",
    top_feat_count: int = 10,
):
    # plot the mean absolute SHAP values for
    # any models
    # NOTE: shap_values should be for the "positive" class,
    # though for now it probably doesn't matter since we have
    # a binary classification with symmetric feature importances
    fig_name = fig_name_stem + ".png"
    fig_source = fig_name_stem + ".csv"
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    mean_abs_shap_values = shap_values.abs.mean(0).values
    mean_abs_shap_values, feature_names = sort_features(
        mean_abs_shap_values, np.array(shap_values.feature_names)
    )
    df = pd.DataFrame()
    df["Feature Name"] = feature_names
    df["Mean Absolute SHAP value"] = mean_abs_shap_values
    df.to_csv(fig_source, index=False)
    feature_names = feature_names[-top_feat_count:]
    mean_abs_shap_values = mean_abs_shap_values[-top_feat_count:]
    y_pos = np.arange(top_feat_count)
    ax.barh(y_pos, mean_abs_shap_values)
    ax.set_title(
        f"Mean Absolute SHAP Value of Top {top_feat_count} Features\n{model_name}"
    )
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_yticks(y_pos, labels=feature_names)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300)
    plt.close()


def plot_shap_beeswarm(
    positive_shap_values,
    model_name: str = "",
    fig_name: str = "feat_shap_beeswarm.png",
    max_display: int = 20,
):
    shap.summary_plot(positive_shap_values, max_display=max_display, show=False)
    plt.title(f"Effect of Top {max_display} Features\n{model_name}")
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()
