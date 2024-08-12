from viral_seq.analysis import feature_importance as fi
import numpy as np
from numpy.testing import assert_array_equal
import matplotlib
from matplotlib.testing.decorators import image_comparison

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


@image_comparison(
    baseline_images=["test_plot_feat_import"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plot_feat_import(tmpdir):
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


@image_comparison(
    baseline_images=["test_plot_feat_import_consensus"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_plot_feat_import_consensus(tmpdir):
    ranked_feature_names = np.array(
        ["Feature 8", "Feature 0", "Feature 3", "Feature 1", "Feature 7", "Feature 2"]
    )
    ranked_feature_counts = np.array([1, 2, 3, 3, 3, 3])
    with tmpdir.as_cwd():
        fi.plot_feat_import_consensus(ranked_feature_names, ranked_feature_counts, 3, 5)
