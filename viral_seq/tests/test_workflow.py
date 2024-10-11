from viral_seq import run_workflow as workflow
from matplotlib.testing.decorators import image_comparison
import numpy as np
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
import shap


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


def test_percent_surface_exposed():
    syn_kmers = ["CAACAAD", "CAACAAD", "FEAGAD", "FEAGAD", "FEAGAD", "FEAGAD", "GACADA"]
    syn_status = ["Yes", "No", "Yes", "Yes", "Yes", "No", "No"]

    out_dict = workflow.percent_surface_exposed(syn_kmers, syn_status)

    assert out_dict["CAACAAD"] == [1, 1]
    assert out_dict["FEAGAD"] == [3, 1]
    assert out_dict["GACADA"] == [0, 1]


def test_fic_plot(tmp_path):
    array2 = [
        "kmer_PC_CDDEEC",
        "kmer_PC_CCGDEA",
        "kmer_PC_CCCFCF",
        "kmer_PC_CCAAACD",
        "kmer_PC_CACDGA",
        "kmer_PC_CFCEDD",
        "kmer_PC_GCECFD",
        "kmer_PC_ECDGDE",
        "kmer_PC_CCACAD",
        "kmer_PC_FECAEA",
    ]
    array1 = np.array([11.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    target_column = "0"
    found_kmers = [
        "CDDEEC",
        "CCGDEA",
        "CCCFCF",
        "CCAAACD",
        "CACDGA",
        "CFCEDD",
        "GCECFD",
        "ECDGDE",
        "CCACAD",
        "FECAEA",
    ]
    surface_exposed_dict = {
        "AFDAEF": [14, 65],
        "CACDGA": [21, 104],
        "CCAAACD": [19, 82],
        "CCACAD": [18, 89],
        "CCCFCF": [23, 106],
        "CCGDEA": [0, 11],
        "CDDEEC": [5, 57],
        "CECCAF": [4, 26],
        "CFCEDD": [11, 41],
        "DFDCCA": [3, 20],
        "DFDCCC": [3, 11],
        "DGACFC": [22, 120],
        "DGDACD": [13, 78],
        "EADAAC": [10, 48],
        "ECDGDE": [18, 93],
        "EGCCAC": [5, 58],
        "FCCGDA": [19, 69],
        "FECAEA": [11, 39],
        "FGACCA": [10, 46],
        "GCECFD": [12, 35],
    }
    not_exposed = [
        "CDDEEC",
        "CCGDEA",
        "",
        "CCAAACD",
        "CACDGA",
        "CFCEDD",
        "GCECFD",
        "ECDGDE",
        "CCACAD",
        "FECAEA",
    ]
    is_exposed = [
        "CDDEEC",
        "",
        "",
        "CCAAACD",
        "CACDGA",
        "CFCEDD",
        "GCECFD",
        "ECDGDE",
        "CCACAD",
        "FECAEA",
    ]

    X, y = make_classification(n_samples=15, n_features=10, random_state=0)
    X = pd.DataFrame(X)
    random_state = 0
    n_folds = 2
    clfr = RandomForestClassifier(
        n_estimators=10000, n_jobs=-1, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=n_folds)
    for fold, (train, test) in enumerate(cv.split(X, y)):
        clfr.fit(X.iloc[train], y[train])

    explainer = shap.Explainer(clfr, seed=random_state)
    shap_values = explainer(X)
    positive_shap_values = shap_values[:, :, 1]

    response_effect, surface_exposed_sign = workflow.FIC_plot(
        array2,
        array1,
        n_folds,
        target_column,
        positive_shap_values,
        found_kmers,
        surface_exposed_dict,
        not_exposed,
        is_exposed,
        paths=[tmp_path],
    )

    response_effect_exp = ["-", "-", "+", "+", "-", "+", "+", "-", "+", "+"]
    surface_exposed_exp = ["+", "-", "x", "+", "+", "+", "+", "+", "+", "+"]

    assert_array_equal(response_effect, response_effect_exp)
    assert_array_equal(surface_exposed_sign, surface_exposed_exp)
