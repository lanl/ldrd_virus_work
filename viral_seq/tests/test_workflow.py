from viral_seq import run_workflow as workflow
from matplotlib.testing.decorators import image_comparison
import numpy as np
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from matplotlib.testing.compare import compare_images


@image_comparison(
    baseline_images=["test_optimization_plotting"],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_optimization_plotting(tmp_path):
    rng = np.random.default_rng(seed=2024)
    data = {
        "Classifier1": rng.uniform(size=30),
        "Classifier2": rng.uniform(size=10),
        "Classifier3": rng.uniform(size=51),
    }
    workflow.optimization_plots(
        data,
        str(tmp_path / "test_optimization_plotting.csv"),
        str(tmp_path / "test_optimization_plotting.png"),
    )


@pytest.mark.parametrize("extract", [True, False])
def test_get_test_features(extract, tmpdir):
    raw_file = files("viral_seq.tests.inputs").joinpath(
        "get_test_features_test_file.csv"
    )
    test_file = str(raw_file)
    X_train = pd.read_csv(
        files("viral_seq.tests.inputs").joinpath("get_test_features_X_train.csv")
    )
    table_loc_test = str(
        files("viral_seq.tests.inputs") / "get_test_features_table_loc_test"
    )
    extract_cookie = raw_file if extract else files("viral_seq.tests") / "fake_file.dat"
    with tmpdir.as_cwd():
        with ExitStack() as stack:
            if extract:
                stack.enter_context(pytest.raises(NameError, match="table_info"))
            X_test, y_test = workflow.get_test_features(
                table_loc_test,
                "X_test.parquet.gzip",
                test_file,
                X_train,
                extract_cookie,
                debug=True,
            )
    if not extract:
        X_expected = pd.read_csv(
            files("viral_seq.tests.expected") / "get_test_features_X_expected.csv"
        )
        y_expected = pd.read_csv(
            files("viral_seq.tests.expected") / "get_test_features_y_expected.csv"
        )["Human Host"]
        assert_frame_equal(X_test, X_expected)
        assert_series_equal(y_test, y_expected)


def test_csv_conversion():
    input_csv = files("viral_seq.data") / "receptor_training.csv"
    postprocessed_df = workflow.csv_conversion(input_csv)
    assert_array_equal(
        postprocessed_df.columns,
        [
            "Species",
            "Accessions",
            "Human Host",
            "Is_Integrin",
            "Is_Sialic_Acid",
            "Is_Both",
        ],
    )
    assert postprocessed_df.shape == (94, 6)
    assert postprocessed_df.sum().Is_Integrin == 45
    assert postprocessed_df.sum().Is_Sialic_Acid == 53
    assert postprocessed_df.sum().Is_Both == 4


def test_label_surface_exposed():
    kmers_list = [
        "CADAFFE",
        "CADAFFE",
        "CCABDAC",
        "CCABDAC",
        "CCABDAC",
        "CCAACDA",
        "CCAACDA",
        "CADAFFE",
        "CADAFFE",
        "ECDGDE",
    ]
    kmers_status = ["Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes"]
    kmers_list_status = list(set(zip(kmers_list, kmers_status)))

    kmers_topN = [
        "kmer_PC_CADAFFE",
        "kmer_AA_CCABDAC",
        "kmer_PC_CCAACDA",
        "kmer_AA_CADAFFE",
        "kmer_PC_ECDGDE",
    ]

    is_exposed_exp = ["CADAFFE", "", "CCAACDA", "CADAFFE", "ECDGDE"]
    not_exposed_exp = ["CADAFFE", "CCABDAC", "", "CADAFFE", ""]
    found_kmers_exp = ["CADAFFE", "CCABDAC", "CCAACDA", "CADAFFE", "ECDGDE"]

    is_exposed, not_exposed, found_kmers = workflow.label_surface_exposed(
        kmers_list_status, kmers_topN
    )

    np.testing.assert_array_equal(is_exposed, is_exposed_exp)
    np.testing.assert_array_equal(not_exposed, not_exposed_exp)
    np.testing.assert_array_equal(found_kmers, found_kmers_exp)


def test_importances_df():
    rng = np.random.default_rng(seed=123)
    importances = rng.uniform(-1, 1, 10)
    train_columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    train_data = np.zeros([10, 10])
    train_fold = pd.DataFrame(train_data, columns=train_columns)
    important_features_exp = ["G", "J", "I", "F", "A", "H", "C", "D", "E", "B"]

    importances_out = workflow.importances_df(importances, train_fold.columns)

    assert importances_out.shape == (10, 3)
    np.testing.assert_array_equal(
        np.array(importances_out["Features"]), important_features_exp
    )

    with pytest.raises(
        ValueError, match="Importances and train features must have same shape."
    ):
        workflow.importances_df(importances[:5], train_fold.columns)
    with pytest.raises(
        ValueError, match="Importances and train features must be a single column."
    ):
        workflow.importances_df(importances, train_fold)


def test_plot_cv_roc(tmp_path):
    rng = np.random.default_rng(seed=123)
    pred_prob = rng.uniform(0, 1, 10)
    true_class = rng.choice([0, 1], size=10)
    data_in = np.stack((pred_prob, true_class))

    workflow.plot_cv_roc([data_in], "Test", tmp_path)
    assert (
        compare_images(
            files("viral_seq.tests.expected") / "ROC_cv_expected.png",
            str(tmp_path / "ROC_Test.png"),
            0.001,
        )
        is None
    )


def test_feature_count_consensus():
    rng = np.random.default_rng(seed=123)
    clfr_importances = rng.uniform(-1, 1, 10)
    shap_importances = rng.uniform(-1, 1, 10)
    train_columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    clfr_importances_df = pd.DataFrame()
    clfr_importances_df["Features"] = train_columns
    clfr_importances_df["Importances"] = clfr_importances
    shap_importances_df = pd.DataFrame()
    shap_importances_df["Features"] = train_columns
    shap_importances_df["Importances"] = shap_importances
    feature_count = pd.DataFrame()
    feature_count["Features"] = train_columns
    feature_count["Counts"] = 0

    feature_count_out_exp = pd.DataFrame()
    feature_count_out_exp["Features"] = train_columns
    feature_count_out_exp["Counts"] = [1, 0, 1, 0, 1, 2, 2, 0, 2, 1]

    feature_count_exp = feature_count.copy()

    clfr_importances_df.sort_values(by=["Importances"], ascending=False, inplace=True)
    clfr_importances_df.reset_index(inplace=True)

    shap_importances_df.sort_values(by=["Importances"], ascending=False, inplace=True)
    shap_importances_df.reset_index(inplace=True)

    feature_count_out = workflow.feature_count_consensus(
        clfr_importances_df, shap_importances_df, feature_count, n_features=5
    )

    assert_frame_equal(feature_count_out, feature_count_out_exp)
    assert_frame_equal(feature_count, feature_count_exp)
