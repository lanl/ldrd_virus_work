from viral_seq.analysis import feature_importance as fi
from viral_seq import run_workflow as workflow
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib
from matplotlib.testing.compare import compare_images
from importlib.resources import files
import shap
import pytest


matplotlib.use("Agg")


def test_sort_features():
    rng = np.random.default_rng(123)
    names = ["Feature " + str(i) for i in range(20)]
    integers = list(rng.integers(0, 10, len(names)))
    floats = list(rng.random(len(names)))
    D1 = {names[i]: integers[i] for i in range(len(names))}
    D2 = {names[i]: floats[i] for i in range(len(names))}
    for D in [D1, D2]:
        sorted_imps, sorted_names = fi.sort_features(
            np.array(list(D.values())), np.array(list(D.keys()))
        )
        assert_array_equal(sorted_imps, sorted(sorted_imps))
        # check name still linked to right score
        result_dict = {
            sorted_names[i]: sorted_imps[i] for i in range(len(sorted_names))
        }
        assert result_dict == D


def test_feature_importance_consensus():
    pos_class_feat_imps = [
        np.array(
            [
                0.13530306,
                0.10153095,
                0.09329931,
                0.2744151,
                0.03632587,
                0.07859788,
                0.05613625,
                0.09979211,
                0.08006325,
                0.04453623,
            ]
        ),  # RFC like
        np.array([47, 44, 29, 49, 24, 22, 24, 26, 19, 6]),  # LGBM like
        np.array(
            [
                [
                    -0.02884951,
                    0.01171798,
                    0.17225616,
                    -0.12776078,
                    0.01019924,
                    0.01629393,
                    0.02708448,
                    0.06069739,
                    0.10540845,
                    0.03175265,
                ],
                [
                    -0.05497881,
                    -0.15289707,
                    -0.03737597,
                    0.08333085,
                    -0.01245506,
                    0.00218519,
                    -0.01960376,
                    -0.16138776,
                    -0.06449641,
                    -0.01352119,
                ],
            ]
        ),  # SHAP like
    ]
    (
        ranked_feature_names,
        ranked_feature_counts,
        num_input_models,
    ) = fi.feature_importance_consensus(
        pos_class_feat_imps, np.array(["Feature " + str(i) for i in range(10)]), 5
    )
    assert_array_equal(
        ranked_feature_names,
        np.array(
            [
                "Feature 8",
                "Feature 0",
                "Feature 3",
                "Feature 1",
                "Feature 7",
                "Feature 2",
            ]
        ),
    )
    assert_array_equal(ranked_feature_counts, np.array([1, 2, 3, 3, 3, 3]))
    assert num_input_models == 3


def test_plot_feat_import(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_feat_import.png"
    )
    sorted_feature_importances = [
        0.03632587,
        0.04453623,
        0.05613625,
        0.07859788,
        0.08006325,
        0.09329931,
        0.09979211,
        0.10153095,
        0.13530306,
        0.2744151,
    ]
    with tmpdir.as_cwd():
        fi.plot_feat_import(
            sorted_feature_importances, ["Feature " + str(i) for i in range(10)], 10
        )
        assert compare_images(expected_plot, "feat_imp.png", 0.001) is None


def test_plot_feat_import_consensus(tmpdir):
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_feat_import_consensus.png"
    )
    ranked_feature_names = np.array(
        ["Feature 8", "Feature 0", "Feature 3", "Feature 1", "Feature 7", "Feature 2"]
    )
    ranked_feature_counts = np.array([1, 2, 3, 3, 3, 3])
    with tmpdir.as_cwd():
        fi.plot_feat_import_consensus(ranked_feature_names, ranked_feature_counts, 3, 5)
        assert compare_images(expected_plot, "feat_imp_consensus.png", 0.001) is None


def test_get_positive_shap_values():
    rng = np.random.default_rng(123)
    cases = []
    # our expected will remain the same, but we'll shape it into different shap_value cases
    expected = rng.uniform(-1.0, 1.0, (5, 2))  # 5 samples, 2 features
    list_case = [-expected, expected]
    cases.append(list_case)
    cases.append(expected)  # xgboost, raw values case
    cases.append(shap.Explanation(values=expected))  # xgboost case
    shap_values = np.moveaxis(np.array(list_case), 0, -1)
    cases.append(shap_values)  # ndim==3, raw values case
    cases.append(shap.Explanation(values=shap_values))  # ndim==3 case
    for case in cases:
        res = fi.get_positive_shap_values(case)
        if isinstance(res, np.ndarray):
            actual = res
        else:
            actual = res.values
        assert_allclose(actual, expected)


def test_plot_shap_meanabs(tmpdir):
    rng = np.random.default_rng(42)
    values = rng.uniform(-1.0, 1.0, (5, 2))  # 5 samples, 2 features
    data = np.copy(values)  # feature values
    rng.shuffle(data)
    feature_names = [f"Feature {i}" for i in range(values.shape[1])]
    explanation = shap.Explanation(
        values=values, data=data, feature_names=feature_names
    )
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_shap_meanabs.png"
    )
    with tmpdir.as_cwd():
        fi.plot_shap_meanabs(explanation, top_feat_count=2)
        assert compare_images(expected_plot, "feat_shap_meanabs.png", 0.001) is None


@pytest.mark.skip(
    reason="SHAP 'summary_plot' expects explanation to have value 'base_values'."
)
@pytest.mark.parametrize("interference", [False, True])
def test_plot_shap_beeswarm(tmpdir, interference):
    rng = np.random.default_rng(1984)
    values = rng.uniform(-1.0, 1.0, (5, 2))  # 5 samples, 2 features
    data = np.copy(values)  # feature values
    rng.shuffle(data)
    explanation = shap.Explanation(values=values, data=data)
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_plot_shap_beeswarm.png"
    )
    # checking BUG caused by not closing optimization plot; see MR !50 https://gitlab.lanl.gov/treddy/ldrd_virus_work/-/merge_requests/50#note_309148
    if interference:
        data2 = {
            "Classifier1": rng.uniform(size=30),
            "Classifier2": rng.uniform(size=10),
            "Classifier3": rng.uniform(size=51),
        }
        workflow.optimization_plots(
            data2,
            "test",
            tmpdir,
        )
    np.random.seed(0)  # beeswarm/summary_plot uses numpy for random
    with tmpdir.as_cwd():
        fi.plot_shap_beeswarm(explanation)
        assert compare_images(expected_plot, "feat_shap_beeswarm.png", 0.001) is None
