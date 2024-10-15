from viral_seq import run_workflow as workflow
import numpy as np
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib.testing.compare import compare_images
from numpy.testing import assert_array_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from matplotlib.testing.compare import compare_images
import shap


def test_optimization_plotting(tmpdir):
    rng = np.random.default_rng(seed=2024)
    data = {
        "Classifier1": rng.uniform(size=30),
        "Classifier2": rng.uniform(size=10),
        "Classifier3": rng.uniform(size=51),
    }
    expected_plot = files("viral_seq.tests.expected").joinpath(
        "test_optimization_plotting.png"
    )
    with tmpdir.as_cwd():
        workflow.optimization_plots(
            data,
            "test",
            tmpdir,
        )
        assert (
            compare_images(expected_plot, "test_optimization_plot.png", 0.001) is None
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


@pytest.mark.parametrize(
    "syn_kmers, mapping_method, mode, expected_dict",
    [
        (
            [
                "kmer_PC_FBAAFF",
                "kmer_PC_AACFAF",
                "kmer_PC_BBACFF",
                "kmer_PC_BAFBBBA",
                "kmer_PC_CAABBA",
                "kmer_PC_AAAA",
                "kmer_AA_ECVGDE",
                "kmer_AA_AAFDAE",
                "kmer_AA_CCAFEE",
                "kmer_AA_CGDCCCA",
                "kmer_AA_FGGCCA",
                "kmer_AA_AAAA",
            ],
            "shen_2007",
            "PC",
            {
                "BBA": {
                    0: "kmer_PC_BBACFF",
                    1: "kmer_PC_BAFBBBA",
                    2: "kmer_PC_CAABBA",
                    3: 3,
                },
                "FBA": {0: "kmer_PC_FBAAFF", 1: None, 2: None, 3: 1},
                "AFFA": {0: None, 1: None, 2: None, 3: 0},
                "AABB": {0: "kmer_PC_CAABBA", 1: None, 2: None, 3: 1},
                "AAC": {0: "kmer_PC_AACFAF", 1: None, 2: None, 3: 1},
                "BAFB": {0: "kmer_PC_BAFBBBA", 1: None, 2: None, 3: 1},
                "AAA": {0: "kmer_PC_AAAA", 1: None, 2: None, 3: 1},
            },
        ),
        (
            [
                "kmer_PC_FBAAFF",
                "kmer_PC_AACFAF",
                "kmer_PC_BBACFF",
                "kmer_PC_BAFBBBA",
                "kmer_PC_CAABBA",
                "kmer_PC_AAAA",
                "kmer_AA_ECVGDE",
                "kmer_AA_AAFDAE",
                "kmer_AA_CCAFEE",
                "kmer_AA_CGDCCCA",
                "kmer_AA_FGGCCA",
                "kmer_AA_AAAA",
            ],
            "shen_2007",
            "AA",
            {
                "CCA": {
                    0: "kmer_AA_CCAFEE",
                    1: "kmer_AA_CGDCCCA",
                    2: "kmer_AA_FGGCCA",
                    3: 3,
                },
                "DCA": {0: None, 1: None, 2: None, 3: 0},
                "GDDA": {0: None, 1: None, 2: None, 3: 0},
                "GGCC": {0: "kmer_AA_FGGCCA", 1: None, 2: None, 3: 1},
                "AAF": {0: "kmer_AA_AAFDAE", 1: None, 2: None, 3: 1},
                "CGDC": {0: "kmer_AA_CGDCCCA", 1: None, 2: None, 3: 1},
                "AAA": {0: "kmer_AA_AAAA", 1: None, 2: None, 3: 1},
            },
        ),
        (
            [
                "kmer_PC_416044",
                "kmer_PC_007404",
                "kmer_PC_110744",
                "kmer_PC_1041110",
                "kmer_PC_700110",
                "kmer_PC_0000",
                "kmer_AA_ECVGDE",
                "kmer_AA_AAFDAE",
                "kmer_AA_CCAFEE",
                "kmer_AA_CGDCCCA",
                "kmer_AA_FGGCCA",
                "kmer_AA_AAAA",
            ],
            "jurgen_schmidt",
            "PC",
            {
                "110": {
                    0: "kmer_PC_110744",
                    1: "kmer_PC_1041110",
                    2: "kmer_PC_700110",
                    3: 3,
                },
                "410": {0: None, 1: None, 2: None, 3: 0},
                "0440": {0: None, 1: None, 2: None, 3: 0},
                "0011": {0: "kmer_PC_700110", 1: None, 2: None, 3: 1},
                "007": {0: "kmer_PC_007404", 1: None, 2: None, 3: 1},
                "1041": {0: "kmer_PC_1041110", 1: None, 2: None, 3: 1},
                "000": {0: "kmer_PC_0000", 1: None, 2: None, 3: 1},
            },
        ),
    ],
)
def test_positive_controls(syn_kmers, mapping_method, mode, expected_dict):
    syn_pos_controls = ["CCA", "DCA", "GDDA", "GGCC", "AAF", "CGDC", "AAA"]

    out_df = workflow.check_positive_controls(
        positive_controls=syn_pos_controls,
        kmers_list=syn_kmers,
        mapping_method=mapping_method,
        mode=mode,
    )
    expected_df = (
        pd.DataFrame.from_dict(expected_dict).replace({np.nan: None}).convert_dtypes()
    )
    assert_frame_equal(out_df, expected_df)


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
    array1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
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
        not_exposed,
        is_exposed,
        paths=[tmp_path],
    )

    response_effect_exp = ["-", "-", "+", "+", "-", "+", "+", "-", "+", "+"]
    surface_exposed_exp = ["+", "-", "x", "+", "+", "+", "+", "+", "+", "+"]

    img_out = compare_images(
        files("viral_seq.tests.expected") / "FIC_expected.png",
        str(tmp_path / "FIC_0.png"),
        50,
    )

    assert img_out is None
    assert_array_equal(response_effect, response_effect_exp)
    assert_array_equal(surface_exposed_sign, surface_exposed_exp)
