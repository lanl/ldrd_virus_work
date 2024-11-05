from viral_seq import run_workflow as workflow
import numpy as np
from importlib.resources import files
from contextlib import ExitStack
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib.testing.compare import compare_images
from numpy.testing import assert_array_equal


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
    "mode, expected, expected_dict",
    [
        (
            "PC",
            'Count of PC-kmer positive controls found in Test kmers:\n{\n "BBA": 3,\n "FBA": 1,\n "AFFA": 0,\n "AABB": 1,\n "AAC": 1,\n "BAFB": 1,\n "AAA": 1\n}\n',
            {
                "BBA": {0: "kmer_PC_BBACFF", 1: "kmer_PC_BAFBBBA", 2: "kmer_PC_CAABBA"},
                "FBA": {0: "kmer_PC_FBAAFF", 1: "empty", 2: "empty"},
                "AFFA": {0: "empty", 1: "empty", 2: "empty"},
                "AABB": {0: "kmer_PC_CAABBA", 1: "empty", 2: "empty"},
                "AAC": {0: "kmer_PC_AACFAF", 1: "empty", 2: "empty"},
                "BAFB": {0: "kmer_PC_BAFBBBA", 1: "empty", 2: "empty"},
                "AAA": {0: "kmer_PC_AAAA", 1: "empty", 2: "empty"},
            },
        ),
        (
            "AA",
            'Count of AA-kmer positive controls found in Test kmers:\n{\n "CCA": 3,\n "DCA": 0,\n "GDDA": 0,\n "GGCC": 1,\n "AAF": 1,\n "CGDC": 1,\n "AAA": 1\n}\n',
            {
                "CCA": {0: "kmer_AA_CCAFEE", 1: "kmer_AA_CGDCCCA", 2: "kmer_AA_FGGCCA"},
                "DCA": {0: "empty", 1: "empty", 2: "empty"},
                "GDDA": {0: "empty", 1: "empty", 2: "empty"},
                "GGCC": {0: "kmer_AA_FGGCCA", 1: "empty", 2: "empty"},
                "AAF": {0: "kmer_AA_AAFDAE", 1: "empty", 2: "empty"},
                "CGDC": {0: "kmer_AA_CGDCCCA", 1: "empty", 2: "empty"},
                "AAA": {0: "kmer_AA_AAAA", 1: "empty", 2: "empty"},
            },
        ),
    ],
)
def test_positive_controls(mode, expected, expected_dict, capsys, tmpdir):
    syn_pos_controls = ["CCA", "DCA", "GDDA", "GGCC", "AAF", "CGDC", "AAA"]

    syn_kmers = [
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
    ]

    workflow.check_positive_controls(
        positive_controls=syn_pos_controls,
        kmers_list=syn_kmers,
        mapping_method="shen_2007",
        input_data="Test",
        mode=mode,
        save_path=tmpdir,
    )

    out_file = tmpdir.join(f"Test_{mode}_kmer_positive_controls.csv")
    captured = capsys.readouterr()
    out_df = pd.read_csv(out_file).fillna("empty")
    out_dict = out_df.to_dict()

    assert captured.out == expected
    assert out_dict == expected_dict
