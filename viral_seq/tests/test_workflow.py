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


@pytest.mark.parametrize(
    "kmers_list, kmers_status, kmers_topN, is_exposed_exp",
    [
        # this test case checks the output of the function when using the
        # 'shen_2007' mapping method. This includes a fix for the bug in
        # issue #93, which was incorrectly mapping the kmer 'kmer_AA_CADAFFE'
        # as surface exposed due to the lack of 'kmer_' prefixes causing double
        # counting of "kmer_status = 'yes'" for both (PC and AA) versions of the kmer
        (
            [
                "kmer_PC_CADAFFE",
                "kmer_PC_CADAFFE",
                "kmer_AA_CCABDAC",
                "kmer_AA_CCABDAC",
                "kmer_AA_CCABDAC",
                "kmer_PC_CCAACDA",
                "kmer_PC_CCAACDA",
                "kmer_AA_CADAFFE",
                "kmer_AA_CADAFFE",
                "kmer_PC_ECDGDE",
            ],
            ["Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes"],
            [
                "kmer_PC_CADAFFE",
                "kmer_AA_CCABDAC",
                "kmer_PC_CCAACDA",
                "kmer_AA_CADAFFE",
                "kmer_PC_ECDGDE",
            ],
            ["kmer_PC_CADAFFE", "", "kmer_PC_CCAACDA", "", "kmer_PC_ECDGDE"],
        ),
        # this test case checks the output of the function when using the 'jurgen_schmidt'
        # mapping method.
        (
            [
                "kmer_PC_01234",
                "kmer_AA_ABCDE",
                "kmer_PC_1234567",
                "kmer_PC_1234567",
                "kmer_AA_HIJKL",
                "kmer_AA_HIJKL",
            ],
            ["Yes", "No", "Yes", "Yes", "No", "No"],
            [
                "kmer_PC_01234",
                "kmer_AA_ABCDE",
                "kmer_PC_1234567",
                "kmer_AA_HIJKL",
            ],
            ["kmer_PC_01234", "", "kmer_PC_1234567", ""],
        ),
    ],
)
def test_label_surface_exposed(kmers_list, kmers_status, kmers_topN, is_exposed_exp):
    kmers_list_status = list(set(zip(kmers_list, kmers_status)))
    is_exposed = workflow.label_surface_exposed(kmers_list_status, kmers_topN)
    np.testing.assert_array_equal(is_exposed, is_exposed_exp)


@pytest.mark.parametrize(
    "syn_kmers, mapping_method, mode, expected_dict",
    [
        (
            [
                "kmer_PC_621166",
                "kmer_PC_112616",
                "kmer_PC_221366",
                "kmer_PC_2162221",
                "kmer_PC_311221",
                "kmer_PC_1111",
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
                "221": {
                    0: "kmer_PC_221366",
                    1: "kmer_PC_2162221",
                    2: "kmer_PC_311221",
                    3: 3,
                },
                "621": {0: "kmer_PC_621166", 1: None, 2: None, 3: 1},
                "1661": {0: None, 1: None, 2: None, 3: 0},
                "1122": {0: "kmer_PC_311221", 1: None, 2: None, 3: 1},
                "113": {0: None, 1: None, 2: None, 3: 0},
                "2162": {0: "kmer_PC_2162221", 1: None, 2: None, 3: 1},
                "111": {0: "kmer_PC_1111", 1: None, 2: None, 3: 1},
            },
        ),
        (
            [
                "kmer_PC_621166",
                "kmer_PC_112616",
                "kmer_PC_221366",
                "kmer_PC_2162221",
                "kmer_PC_311221",
                "kmer_PC_1111",
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
        "kmer_PC_CDDEEC": 42.86,
        "kmer_PC_CCGDEA": 0.00,
        "kmer_PC_CCCFCF": 0.00,
        "kmer_PC_CCAAACD": 21.15,
        "kmer_PC_CACDGA": 13.04,
        "kmer_PC_CFCEDD": 25.53,
        "kmer_PC_GCECFD": 17.86,
        "kmer_PC_ECDGDE": 100.0,
        "kmer_PC_CCACAD": 17.24,
        "kmer_PC_FECAEA": 14.29,
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
                "kmer_PC_CAACAAD",
                "kmer_PC_CAACAAD",
                "kmer_PC_FEAGAD",
                "kmer_PC_FEAGAD",
                "kmer_PC_FEAGAD",
                "kmer_PC_FEAGAD",
                "kmer_PC_GACADA",
            ],
            ["Yes", "No", "Yes", "Yes", "Yes", "No", "No"],
            [50.0, 75.0, 0.0],
        ],
        [
            [
                "kmer_PC_0122345",
                "kmer_PC_0122345",
                "kmer_PC_0122345",
                "kmer_PC_0122345",
                "kmer_PC_741065",
                "kmer_PC_741065",
                "kmer_PC_741065",
            ],
            ["Yes", "Yes", "Yes", "Yes", "No", "No", "No"],
            [100.0, 0.0],
        ],
        [
            [
                "kmer_AA_RVDAQL",
                "kmer_AA_RVDAQL",
                "kmer_AA_RVDAQL",
                "kmer_AA_TYVWRCP",
                "kmer_AA_ILGNMCS",
                "kmer_AA_ILGNMCS",
                "kmer_AA_ILGNMCS",
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
        # this test checks that the correct virus-protein pairs are found
        # when using the shen_2007 mapping scheme.
        (
            ["NC_001563.2"],
            5,
            ["WNV"] * 5,
            [
                "kmer_PC_164156",
                "kmer_PC_633741",
                "kmer_PC_514113",
                "kmer_PC_471363",
                "kmer_AA_LVFGGIT",
            ],
            [
                "envelope protein E",
                "nonstructural protein NS3",
                "envelope protein E",
                "RNA-dependent RNA polymerase NS5",
                "nonstructural protein NS2A",
            ],
            "shen_2007",
        ),
        # this test checks that the correct virus-protein products are found
        # when using the "jurgen_schmidt" mapping scheme
        (
            ["NC_001563.2"],
            1,
            ["WNV"],
            ["kmer_AA_LVFGGIT"],
            ["nonstructural protein NS2A"],
            "jurgen_schmidt",
        ),
        # this test checks that nothing is returned for accession records
        # that do not contain instances of the topN kmers
        (
            ["AC_000008.1"],
            0,
            [],
            [],
            [],
            "jurgen_schmidt",
        ),
        # this test checks that the function finds single polyprotein gene products
        (
            ["NC_039210.1"],
            1,
            ["FMDV"],
            ["kmer_PC_300840"],
            ["polyprotein"],
            "jurgen_schmidt",
        ),
        # this test checks the patch for the bug fix in !122.
        # the genbank file for Ross River Virus contains two
        # polyproteins, i.e., "non-structural" and "structural"
        # as well as mature protein products
        # This test asserts that only the mature protein
        # products are found when calling get_kmer_info
        (
            ["NC_075016.1"],
            2,
            ["Ross River virus (RR)", "Ross River virus (RR)"],
            ["kmer_PC_60837226", "kmer_PC_45724566"],
            ["nsP3 protein", "E2 protein"],
            "jurgen_schmidt",
        ),
        # this case utilizes the PC kmer '6203664', which is the longest kmer shared between the viral sequences
        # stored in the multi-cassette accession for PHV, which recapitulates the bug described in issue #107
        # TODO: the observance of both precursor ('G1 and G2 proteins') and mature
        # protein products('envelope surface glycoprotein G2') in this accession has
        # been noted in #102 and needs to be dealt with in a separate MR
        (
            ["NC_038939.1", "NC_038940.1"],
            3,
            ["Prospect hill hantavirus (PHV)"] * 3,
            ["kmer_PC_6203664"] * 3,
            [
                "RNA-dependent RNA polymerase",
                "G1 and G2 proteins",
                "envelope surface glycoprotein G1",
            ],
            "jurgen_schmidt",
        ),
    ],
)
def test_get_kmer_info(
    accession,
    exp,
    exp_viruses,
    exp_kmers,
    exp_proteins,
    mapping_method,
):
    this_cache = files("viral_seq.tests") / "cache_test"
    cache_str = str(this_cache.resolve())  # type: ignore[attr-defined]
    records = sp.load_from_cache(
        accessions=accession, cache=cache_str, verbose=True, filter=False
    )
    # 'data_table' is used to look up the information associated with the
    # viral protein in which the kmer feature is found. Only those entries that
    # contain the appropriate accessions are necessary for the testing data_table
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
            10: "Ross River virus (RR)",
            11: "Prospect hill hantavirus (PHV)",
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
            10: "NC_075016.1",
            11: "NC_038939.1 NC_038940.1 NC_038938.1",
        },
    }
    tbl = pd.DataFrame(data_table)

    kmers = [
        "ADMAHD",
        "DFFKSG",
        "NGTGGI",
        "NGAPEA",
        "SRGLDP",
        "MYGQLIE",  # using the jurgen_schmidt mapping scheme, this translates to '6203664' and is the longest PC kmer shared between the two accessions corresponding to PHV
        "YDTIPI",
        "VGPNFSTV",  # this AA kmer is found in RR "non-structural polyprotein" and "nsP3 protein"
        "DKFSERII",  # this AA kemr is found in RR "structual polyprotein" and "E2 protein"
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


def test_percent_exposed_error():
    rng = np.random.default_rng(seed=123)

    kmer_names = rng.integers(0, 9, size=10)
    kmer_names = [str(n) for n in kmer_names]

    syn_status = rng.choice(["yes", "no"], size=10)

    with pytest.raises(ValueError, match="kmer feature name missing prefix."):
        workflow.percent_surface_exposed(kmer_names, syn_status)


@pytest.mark.parametrize(
    "pos_con_dict, kmer_prefix, mapping_method, dataset_name, exp_output",
    [
        (
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
            "PC",
            "jurgen_schmidt",
            "Test",
            "Count of Positive Control PC k-mers in Test Dataset:\n 110  410  0440 0011 007 1041 000\n  3  0.0   0.0    1   1    1   1\n",
        ),
    ],
)
def test_print_pos_con(
    capsys, tmpdir, pos_con_dict, kmer_prefix, mapping_method, dataset_name, exp_output
):
    pos_con_df = pd.DataFrame(pos_con_dict)
    with tmpdir.as_cwd():
        workflow.print_pos_con(pos_con_df, kmer_prefix, mapping_method, dataset_name)

        fname = f"{dataset_name}_data_{kmer_prefix}_kmer_positive_controls_{mapping_method}.csv"
        actual = pd.read_csv(fname)
        assert actual.shape == (4, 7)
        assert list(actual.iloc[:, 0]) == [
            "kmer_PC_110744",
            "kmer_PC_1041110",
            "kmer_PC_700110",
            "3",
        ]

        captured = capsys.readouterr()
        assert captured.out == exp_output


def test_importances_df():
    np.random.seed(123)
    importances = np.random.uniform(-1, 1, 10)
    train_columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    train_data = np.zeros([10, 10])
    train_fold = pd.DataFrame(train_data, columns=train_columns)

    importances_out = workflow.importances_df(importances, train_fold.columns)

    assert importances_out.shape == (10, 3)
