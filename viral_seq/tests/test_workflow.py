from viral_seq import run_workflow as workflow
import numpy as np
from importlib.resources import files
from contextlib import ExitStack, nullcontext
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from matplotlib.testing.compare import compare_images
from numpy.testing import assert_array_equal, assert_allclose
from viral_seq.analysis import spillover_predict as sp
from viral_seq.analysis import get_features
from numpy.testing import assert_array_equal
from matplotlib.testing.compare import compare_images


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

    array1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.7, 1.8, 1.9, 2.0])
    target_column = "Is_Integrin"

    response_effect_sign = ["+", "-", "+", "+", "+", "+", "+", "-", "+", "-"]
    exposure_status_sign = ["+", "-", "-", "+", "+", "+", "+", "+", "+", "+"]
    surface_exposed_dict = {
        "CDDEEC": 42.86,
        "CCGDEA": 0.00,
        "CCCFCF": 0.00,
        "CCAAACD": 21.15,
        "CACDGA": 13.04,
        "CFCEDD": 25.53,
        "GCECFD": 17.86,
        "ECDGDE": 100.0,
        "CCACAD": 17.24,
        "FECAEA": 14.29,
    }

    n_folds = 2

    workflow.FIC_plot(
        array2,
        array1,
        n_folds,
        target_column,
        exposure_status_sign,
        response_effect_sign,
        surface_exposed_dict,
        tmp_path,
    )

    assert (
        compare_images(
            files("viral_seq.tests.expected") / "FIC_expected.png",
            str(tmp_path / "FIC_Is_Integrin.png"),
            0.001,
        )
        is None
    )


@pytest.mark.parametrize(
    "constant, not_exposed_idx, surface_exposed_exp",
    [
        (
            False,
            [1],
            ["+", "-", "+", "+", "+", "+", "+", "+", "+", "+"],
        ),
        (
            False,
            list(range(10)),
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
        ),
        (
            True,
            [1],
            ["+", "-", "+", "+", "+", "+", "+", "+", "+", "+"],
        ),
    ],
)
def test_feature_sign(
    constant,
    not_exposed_idx,
    surface_exposed_exp,
):
    response_effect_exp = ["+", "-", "+", "+", "+", "+", "+", "-", "+", "-"]
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
    is_exposed = [
        s if i not in not_exposed_idx else "" for i, s in enumerate(found_kmers)
    ]

    rng = np.random.default_rng(seed=123)
    syn_shap_values = rng.uniform(-1, 1, (10, 10))
    syn_data = rng.choice([0, 1], size=[10, 10])

    if constant:
        # modify shap value and data arrays to
        # account for nan pearson-r calculation case
        syn_shap_values[:, -1] = 0.0
        syn_data[:, -1] = 0

    surface_exposed_out, response_effect_out = workflow.feature_signs(
        is_exposed, syn_shap_values, syn_data
    )

    assert_array_equal(response_effect_out, response_effect_exp)
    assert_array_equal(surface_exposed_out, surface_exposed_exp)


@pytest.mark.parametrize(
    "syn_kmers, syn_status, percent_values",
    (
        [
            [
                "CAACAAD",
                "CAACAAD",
                "FEAGAD",
                "FEAGAD",
                "FEAGAD",
                "FEAGAD",
                "GACADA",
            ],
            ["Yes", "No", "Yes", "Yes", "Yes", "No", "No"],
            [50.0, 75.0, 0.0],
        ],
        [
            [
                "0122345",
                "0122345",
                "0122345",
                "0122345",
                "741065",
                "741065",
                "741065",
            ],
            ["Yes", "Yes", "Yes", "Yes", "No", "No", "No"],
            [100.0, 0.0],
        ],
        [
            [
                "RVDAQL",
                "RVDAQL",
                "RVDAQL",
                "TYVWRCP",
                "ILGNMCS",
                "ILGNMCS",
                "ILGNMCS",
            ],
            ["No", "No", "Yes", "No", "No", "Yes", "No"],
            [33.333333, 0.0, 33.333333],
        ],
        [
            [
                "kmer_AA_RVDAQL",
                "kmer_AA_RVDAQL",
                "kmer_AA_ACGAGD",
                "kmer_PC_ACGAGD",
                "kmer_PC_0122345",
                "kmer_PC_0122345",
                "kmer_PC_741065",
                "kmer_AA_ILGNMCS",
                "kmer_AA_ILGNMCS",
            ],
            ["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No"],
            [0.0, 0.0, 0.0, 50.0, 0.0, 50.0],
        ],
    ),
)
def test_percent_surface_exposed(syn_kmers, syn_status, percent_values):
    out_dict = workflow.percent_surface_exposed(syn_kmers, syn_status)

    assert_allclose(list(out_dict.values()), percent_values)


@pytest.mark.parametrize(
    "kmer_features, kmer_range, exp",
    (
        [
            ["kmer_PC_0123456", "kmer_AA_VLYWG", "kmer_PC_CBBA", "kmer_PC_012345678"],
            "10-10",
            pytest.raises(ValueError, match="k-mer feature lengths"),
        ],
        [
            ["kmer_PC_01234", "kmer_AA_VLYW", "kmer_PC_CBBA", "kmer_PC_5678901"],
            "4-7",
            nullcontext(0),
        ],
        [
            ["kmer_PC_0123456", "kmer_AA_VLYWG", "kmer_PC_CBBA", "kmer_PC_012345678"],
            "10",
            pytest.raises(ValueError, match="k-mer feature lengths"),
        ],
    ),
)
def test_check_kmer_feature_lengths(kmer_features, kmer_range, exp):
    with exp:
        workflow.check_kmer_feature_lengths(kmer_features, kmer_range)


@pytest.mark.parametrize(
    "accession, exp, exp_viruses, exp_kmers, exp_proteins, mapping_method",
    [
        (
            "NC_001563.2",
            12,
            ["WNV"] * 12,
            [
                "AFDAEF",
                "AFDAEF",
                "AFDAEF",
                "FCCGDA",
                "FCCGDA",
                "EADAAC",
                "EADAAC",
                "EADAAC",
                "DGACFC",
                "DGACFC",
                "LVFGGIT",
                "LVFGGIT",
            ],
            [
                "polyprotein",
                "envelope protein E",
                "truncated polyprotein NS1 prime",
                "polyprotein",
                "nonstructural protein NS3",
                "polyprotein",
                "envelope protein E",
                "truncated polyprotein NS1 prime",
                "polyprotein",
                "RNA-dependent RNA polymerase NS5",
                "polyprotein",
                "nonstructural protein NS2A",
            ],
            "shen_2007",
        ),
        (
            "NC_001563.2",
            2,
            ["WNV"] * 2,
            ["LVFGGIT", "LVFGGIT"],
            ["polyprotein", "nonstructural protein NS2A"],
            "jurgen_schmidt",
        ),
        (
            "AC_000008.1",
            0,
            [],
            [],
            [],
            "jurgen_schmidt",
        ),
    ],
)
def test_get_kmer_info(
    accession: str,
    exp: int,
    exp_viruses: list[str],
    exp_kmers: list[str],
    exp_proteins: list[str],
    mapping_method: str,
):
    this_cache = files("viral_seq.tests") / "cache_test"
    cache_str = str(this_cache.resolve())  # type: ignore[attr-defined]
    records = sp.load_from_cache(
        accessions=[accession], cache=cache_str, verbose=True, filter=False
    )
    data_table = {
        "Species": {
            0: "hMPV",
            1: "MERS-CoV",
            2: "FMDV",
            3: "influenza_A_H1N1",
            4: "HSV-2",
            5: "type_3_reovirus",
            6: "WNV",
            7: "type_1_reovirus",
            8: "JEV",
            9: "BKPyV human polyomavirus",
        },
        "Accessions": {
            0: "NC_039199.1",
            1: "NC_019843.3",
            2: "NC_039210.1",
            3: "NC_026438.1 NC_026435.1 NC_026437.1 NC_026433.1 NC_026436.1 NC_026434.1 NC_026432.1 NC_026431.1",
            4: "NC_001798.2",
            5: "NC_077846.1 NC_077845.1 NC_077844.1 NC_077843.1 NC_077842.1 NC_077841.1 NC_077840.1 NC_077839.1 NC_077838.1 NC_077837.1",
            6: "NC_001563.2",
            7: "MW198704.1 MW198707.1 MW198708.1 MW198709.1 MW198710.1 MW198711.1 MW198712.1 MW198713.1 MW198705.1 MW198706.1",
            8: "NC_001437.1",
            9: "NC_001538.1",
        },
    }
    tbl = pd.DataFrame(data_table)

    kmers = [
        "ADMAHD",
        "DFFKSG",
        "HKFLVP",
        "NGTGGI",
        "MRTAPT",
        "SRGLDP",
        "YDTIPI",
        "DRGIFV",
        "MDSIPG",
        "FHIPGE",
        "LVFGGIT",
        "IPKMNV",
        "LIPDIT",
        "FLAGVPT",
        "FALMKV",
        "LDIHMY",
        "KFHFDT",
        "HLTKTW",
        "LIAPGT",
        "DHIAQV",
    ]

    new_kmers = []
    for i, topN_kmer in enumerate(kmers):
        if i < len(kmers) / 2:
            topN_kmer_PC = [
                get_features.aa_map(s, method=mapping_method) for s in topN_kmer
            ]
            new_kmers.append("kmer_PC_" + "".join(topN_kmer_PC))
        else:
            new_kmers.append("kmer_AA_" + topN_kmer)
    data_in = workflow.kmer_data(mapping_method, new_kmers)

    viruses, kmers, protein_name = workflow.get_kmer_info(
        data_in, records, tbl, mapping_method
    )

    assert_array_equal(viruses, exp_viruses)
    assert_array_equal(kmers, exp_kmers)
    assert_array_equal(protein_name, exp_proteins)
    assert len(viruses) == len(kmers) == len(protein_name) == exp


@pytest.mark.parametrize(
    "accession, kmer, mapping_method, mismatch_method",
    [
        (
            "NC_001563.2",
            ["kmer_PC_GADAGA"],
            "shen_2007",
            "jurgen_schmidt",
        ),
        (
            "NC_001563.2",
            ["kmer_PC_012"],
            "jurgen_schmidt",
            "shen_2007",
        ),
    ],
)
def test_get_kmer_info_error(
    accession: str, kmer: list[str], mapping_method: str, mismatch_method: str
):
    this_cache = files("viral_seq.tests") / "cache_test"
    cache_str = str(this_cache.resolve())  # type: ignore[attr-defined]
    records = sp.load_from_cache(
        accessions=[accession], cache=cache_str, verbose=True, filter=False
    )
    data_table = {
        "Species": {
            0: "hMPV",
        },
        "Accessions": {
            0: "NC_039199.1",
        },
    }
    tbl = pd.DataFrame(data_table)
    data_in = workflow.kmer_data(mapping_method, kmer)
    with pytest.raises(ValueError, match="kmer mapping method does not match"):
        _, _, _ = workflow.get_kmer_info(data_in, records, tbl, mismatch_method)


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


<<<<<<< HEAD
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
=======
def test_plot_shap_consensus(tmp_path):
    rng = np.random.default_rng(seed=123)
    syn_shap_values = rng.uniform(-1, 1, (10, 10))
    syn_data = rng.choice([0, 1], size=[10, 10])
    syn_features = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    syn_df = pd.DataFrame(syn_data, columns=syn_features)

    np.random.seed(123)
    workflow.plot_shap_consensus(syn_shap_values, syn_df, "Test", tmp_path)
    assert (
        compare_images(
            files("viral_seq.tests.expected") / "SHAP_consensus_exp.png",
            str(tmp_path / "SHAP_Test.png"),
>>>>>>> 32edbfd (ENH: Plot shap consensus across multiple cv folds)
            0.001,
        )
        is None
    )
<<<<<<< HEAD


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
=======
>>>>>>> 32edbfd (ENH: Plot shap consensus across multiple cv folds)
