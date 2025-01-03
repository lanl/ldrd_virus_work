from viral_seq.analysis import dtra_utils
import pandas as pd
from numpy.testing import assert_array_equal
from importlib.resources import files
from viral_seq.analysis.spillover_predict import _append_recs
from viral_seq.analysis.get_features import get_kmers, KmerData
import pytest
from pandas.testing import assert_frame_equal
import os
import string


def test_get_surface_exposure_status():
    # make lists of virus and protein names
    viruses = ["virus_1", "virus_2", "virus_2", "virus_3"]
    proteins = ["protein_1", "protein_1", "protein_2", "protein_2"]
    # make a dataframe of surface exposure status containing the columns ``virus_names`` and ``protein_names`` and ``surface_exposed_status``
    surface_exposed_df = pd.DataFrame(
        {
            "virus_names": [
                "virus_1",
                "virus_1",
                "virus_2",
                "virus_2",
                "virus_3",
                "virus_3",
            ],
            "protein_names": ["protein_1", "protein_2"] * 3,
            "surface_exposed_status": ["yes", "no"] * 3,
        }
    )

    exp_out = ["yes", "yes", "no", "no"]

    surface_exposed_list = dtra_utils.get_surface_exposure_status(
        viruses, proteins, surface_exposed_df
    )

    assert_array_equal(surface_exposed_list, exp_out)


def test_merge_convert_tbl():
    igsf_data = files("viral_seq.data") / "igsf_training.csv"
    input_csv = files("viral_seq.data") / "receptor_training.csv"

    merged_df = dtra_utils.merge_tables(input_csv, igsf_data)
    converted_df = dtra_utils.convert_merged_tbl(merged_df)

    assert merged_df.shape == (118, 8)
    assert converted_df.shape == (118, 10)
    assert converted_df.IN.sum() == 45
    assert converted_df.SA.sum() == 53
    assert converted_df.SA_IG.sum() == 6
    assert converted_df.IN_IG.sum() == 7
    assert converted_df.IN_SA.sum() == 4
    assert converted_df.IN_SA_IG.sum() == 2
    assert converted_df.IG.sum() == 35


# TODO: re-parameterize this test using hypothesis (issue #140)
@pytest.mark.parametrize(
    "kmer_matches, syn_topN, kmer_matches_exp, topN_saved_exp, mapping_methods",
    [
        (
            # this case tests that the function produces matches when two mapping methods are compared
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                    "1": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                },
                {
                    "0": ["kmer_PC_" + str(x + 1) for x in range(3)],
                    "1": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_S"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
                {
                    "0": ["kmer_PC_" + str(x + 1) for x in range(3)],
                },
            ],
            {
                "jurgen_schmidt": {0: "kmer_PC_0", 1: "kmer_PC_1"},
                "shen_2007": {0: "kmer_PC_1", 1: "kmer_PC_2"},
                "matching AA kmer 0": {0: "kmer_AA_G", 1: "kmer_AA_C"},
            },
            {
                "jurgen_schmidt": {
                    "kmer_PC_0": {0: "kmer_AA_G"},
                    "kmer_PC_1": {0: "kmer_AA_C"},
                    "kmer_PC_2": {0: "kmer_AA_Y"},
                },
                "shen_2007": {
                    "kmer_PC_1": {0: "kmer_AA_G"},
                    "kmer_PC_2": {0: "kmer_AA_C"},
                    "kmer_PC_3": {0: "kmer_AA_S"},
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # this test case checks that the correct values are returned when
        # the topN kmers contains kmer strings have len > 1
        (
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)] + ["kmer_PC_00"] * 2,
                    "1": [
                        "kmer_AA_G",
                        "kmer_AA_C",
                        "kmer_AA_Y",
                        "kmer_AA_AG",
                        "kmer_AA_GA",
                    ],
                },
                {
                    "0": ["kmer_PC_" + str(x + 1) for x in range(3)]
                    + ["kmer_PC_11"] * 2,
                    "1": [
                        "kmer_AA_G",
                        "kmer_AA_C",
                        "kmer_AA_S",
                        "kmer_AA_AG",
                        "kmer_AA_GA",
                    ],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)] + ["kmer_PC_00"],
                },
                {
                    "0": ["kmer_PC_" + str(x + 1) for x in range(3)] + ["kmer_PC_11"],
                },
            ],
            {
                "jurgen_schmidt": {0: "kmer_PC_0", 1: "kmer_PC_00", 2: "kmer_PC_1"},
                "shen_2007": {0: "kmer_PC_1", 1: "kmer_PC_11", 2: "kmer_PC_2"},
                "matching AA kmer 0": {0: "kmer_AA_G", 1: "kmer_AA_AG", 2: "kmer_AA_C"},
                "matching AA kmer 1": {0: None, 1: "kmer_AA_GA", 2: None},
            },
            {
                "jurgen_schmidt": {
                    "kmer_PC_0": {0: "kmer_AA_G", 1: None},
                    "kmer_PC_00": {0: "kmer_AA_AG", 1: "kmer_AA_GA"},
                    "kmer_PC_1": {0: "kmer_AA_C", 1: None},
                    "kmer_PC_2": {0: "kmer_AA_Y", 1: None},
                },
                "shen_2007": {
                    "kmer_PC_1": {0: "kmer_AA_G", 1: None},
                    "kmer_PC_11": {0: "kmer_AA_AG", 1: "kmer_AA_GA"},
                    "kmer_PC_2": {0: "kmer_AA_C", 1: None},
                    "kmer_PC_3": {0: "kmer_AA_S", 1: None},
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # this test case checks that the correct matches are returned when
        # the PC kmers match between schemes, but they map to different AA kmers
        (
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                    "1": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                },
                {
                    "0": ["kmer_PC_4", "kmer_PC_2", "kmer_PC_1"],
                    "1": ["kmer_AA_Y", "kmer_AA_C", "kmer_AA_A"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
                {
                    "0": ["kmer_PC_4", "kmer_PC_2", "kmer_PC_1"],
                },
            ],
            {
                "jurgen_schmidt": {0: "kmer_PC_1", 1: "kmer_PC_2"},
                "shen_2007": {0: "kmer_PC_2", 1: "kmer_PC_4"},
                "matching AA kmer 0": {0: "kmer_AA_C", 1: "kmer_AA_Y"},
            },
            {
                "jurgen_schmidt": {
                    "kmer_PC_0": {0: "kmer_AA_G"},
                    "kmer_PC_1": {0: "kmer_AA_C"},
                    "kmer_PC_2": {0: "kmer_AA_Y"},
                },
                "shen_2007": {
                    "kmer_PC_1": {0: "kmer_AA_A"},
                    "kmer_PC_2": {0: "kmer_AA_C"},
                    "kmer_PC_4": {0: "kmer_AA_Y"},
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # this case tests that nothing is returned when only a single mapping method is used
        # to generate AA kmer matches.
        (
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                    "1": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
            ],
            None,
            {
                "jurgen_schmidt": {
                    "kmer_PC_0": {0: "kmer_AA_G"},
                    "kmer_PC_1": {0: "kmer_AA_C"},
                    "kmer_PC_2": {0: "kmer_AA_Y"},
                }
            },
            ["jurgen_schmidt"],
        ),
        # test when a pair of PC kmers matches through more than one AA kmer
        (
            [
                {
                    "0": ["kmer_PC_1", "kmer_PC_1", "kmer_PC_2"],
                    "1": ["kmer_AA_A", "kmer_AA_C", "kmer_AA_Y"],
                },
                {
                    "0": ["kmer_PC_1", "kmer_PC_1", "kmer_PC_1"],
                    "1": ["kmer_AA_A", "kmer_AA_C", "kmer_AA_Y"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
                {
                    "0": ["kmer_PC_1"],
                },
            ],
            {
                "jurgen_schmidt": {0: "kmer_PC_1", 1: "kmer_PC_2"},
                "shen_2007": {0: "kmer_PC_1", 1: "kmer_PC_1"},
                "matching AA kmer 0": {0: "kmer_AA_A", 1: "kmer_AA_Y"},
                "matching AA kmer 1": {0: "kmer_AA_C", 1: None},
            },
            {
                "jurgen_schmidt": {
                    "kmer_PC_1": {0: "kmer_AA_A", 1: "kmer_AA_C"},
                    "kmer_PC_2": {0: "kmer_AA_Y", 1: None},
                },
                "shen_2007": {
                    "kmer_PC_1": {0: "kmer_AA_A", 1: "kmer_AA_C", 2: "kmer_AA_Y"}
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # test when a PC kmer from one scheme matches with two different PC kmers of the other scheme
        (
            [
                {
                    "0": ["kmer_PC_1", "kmer_PC_1", "kmer_PC_2"],
                    "1": ["kmer_AA_A", "kmer_AA_C", "kmer_AA_Y"],
                },
                {
                    "0": ["kmer_PC_1", "kmer_PC_2", "kmer_PC_2"],
                    "1": ["kmer_AA_A", "kmer_AA_C", "kmer_AA_Y"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
            ],
            {
                "jurgen_schmidt": {0: "kmer_PC_1", 1: "kmer_PC_1", 2: "kmer_PC_2"},
                "shen_2007": {0: "kmer_PC_1", 1: "kmer_PC_2", 2: "kmer_PC_2"},
                "matching AA kmer 0": {0: "kmer_AA_A", 1: "kmer_AA_C", 2: "kmer_AA_Y"},
            },
            {
                "jurgen_schmidt": {
                    "kmer_PC_1": {0: "kmer_AA_A", 1: "kmer_AA_C"},
                    "kmer_PC_2": {0: "kmer_AA_Y", 1: None},
                },
                "shen_2007": {
                    "kmer_PC_1": {0: "kmer_AA_A", 1: None},
                    "kmer_PC_2": {0: "kmer_AA_C", 1: "kmer_AA_Y"},
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # test support for AA kmers in the topN kmers
        (
            [
                {
                    "0": ["kmer_PC_1"],
                    "1": ["kmer_AA_A"],
                },
                {
                    "0": ["kmer_PC_2"],
                    "1": ["kmer_AA_C"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)] + ["kmer_AA_A"],
                },
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)] + ["kmer_AA_A"],
                },
            ],
            {
                "jurgen_schmidt": {0: "kmer_AA_A", 1: "kmer_PC_1"},
                "shen_2007": {0: "kmer_AA_A", 1: "kmer_AA_A"},
                "matching AA kmer 0": {0: "kmer_AA_A", 1: "kmer_AA_A"},
            },
            {
                "jurgen_schmidt": {
                    "kmer_AA_A": {0: "kmer_AA_A"},
                    "kmer_PC_1": {0: "kmer_AA_A"},
                },
                "shen_2007": {
                    "kmer_AA_A": {0: "kmer_AA_A"},
                    "kmer_PC_2": {0: "kmer_AA_C"},
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
    ],
)
def test_match_kmers(
    tmpdir, kmer_matches, syn_topN, kmer_matches_exp, topN_saved_exp, mapping_methods
):
    """test that matching AA-kmers are found between the two mapping methods with different PC-kmers"""

    # make and save a temporary file containing the matching kmer dataframe
    syn_topN_df = [pd.DataFrame(x) for x in syn_topN]
    for i, mm in enumerate(mapping_methods):
        pd.DataFrame(kmer_matches[i]).to_parquet(
            f"{tmpdir}/kmer_maps_{mapping_methods[i]}.parquet.gzip"
        )
    # run the function to generate the matches
    with tmpdir.as_cwd():
        kmer_matches_out = dtra_utils.match_kmers(syn_topN_df, mapping_methods, tmpdir)
    if kmer_matches_out is None:
        kmer_matches_out = pd.DataFrame(kmer_matches_out)
    kmer_matches_exp_df = pd.DataFrame(kmer_matches_exp)
    # assert that the output looks as expected and that a csv file was generated containing the dataframe
    assert_frame_equal(kmer_matches_out, kmer_matches_exp_df)
    for mm in mapping_methods:
        topN_saved_exp_mm = pd.DataFrame(topN_saved_exp[mm])
        topN_saved = pd.read_csv(
            os.path.join(tmpdir, f"topN_PC_AA_kmer_mappings_{mm}.csv")
        )
        assert_frame_equal(topN_saved, topN_saved_exp_mm)


@pytest.mark.parametrize(
    "kmer_matches, syn_topN, mapping_method, output",
    [
        # this test case checks that the function returns the correct string output for
        # comparing kmer matches when the workflow has not been run using both mapping methods
        (
            None,
            None,
            ["jurgen_schmidt", "shen_2007"],
            "Must run workflow using both mapping methods before performing kmer mapping.",
        ),
        # this test case checks that the function returns the correct string output for comparing
        # kmer matches when there ARE NO AA matches found between PC mapping methods
        (
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                    "1": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                },
                {
                    "0": ["kmer_PC_" + str(x + 3) for x in range(3)],
                    "1": ["kmer_AA_F", "kmer_AA_M", "kmer_AA_H"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
                {
                    "0": ["kmer_PC_" + str(x + 3) for x in range(3)],
                },
            ],
            ["jurgen_schmidt", "shen_2007"],
            "No matching AA kmers found between PC mapping schemes in TopN.",
        ),
        # this test case checks that the function returns the correct string output for comparing
        # mapping methods when there ARE AA matches found between PC mapping methods
        (
            [
                {
                    "0": ["kmer_PC_7", "kmer_PC_1", "kmer_PC_5"],
                    "1": ["kmer_AA_F", "kmer_AA_C", "kmer_AA_K"],
                },
                {
                    "0": ["kmer_PC_4", "kmer_PC_2", "kmer_PC_7"],
                    "1": ["kmer_AA_F", "kmer_AA_C", "kmer_AA_K"],
                },
            ],
            [
                {
                    "0": ["kmer_PC_7", "kmer_PC_1", "kmer_PC_5"],
                },
                {
                    "0": ["kmer_PC_4", "kmer_PC_2", "kmer_PC_7"],
                },
            ],
            ["jurgen_schmidt", "shen_2007"],
            "Matching AA kmers between mapping methods:\n jurgen_schmidt shen_2007 matching AA kmer 0\n     kmer_PC_1 kmer_PC_2          kmer_AA_C\n     kmer_PC_5 kmer_PC_7          kmer_AA_K\n     kmer_PC_7 kmer_PC_4          kmer_AA_F",
        ),
    ],
)
def test_find_matching_kmers(tmpdir, kmer_matches, syn_topN, mapping_method, output):
    """
    test that the correct output is returned from ``find_matching_kmers`` for the cases where:
        1. the workflow has not been run with both mapping methods
        2. no matching kmers are found between the two mapping methods
        3. matching kmers are found between the two mapping methods
    """
    with tmpdir.as_cwd():
        if kmer_matches is not None:
            os.mkdir("kmer_maps")
            for i, mm in enumerate(mapping_method):
                syn_topN_df = pd.DataFrame(syn_topN[i])
                syn_topN_df.to_parquet(f"topN_kmers_test_{mm}.parquet.gzip")
                pd.DataFrame(kmer_matches[i]).to_parquet(
                    f"kmer_maps/kmer_maps_{mm}.parquet.gzip"
                )
        result = dtra_utils.find_matching_kmers(
            target_column="test", mapping_methods=mapping_method
        )
        assert result == output

    if result.startswith("Matching AA kmers"):
        assert os.path.exists(
            os.path.join(
                tmpdir, f"{mapping_method[0]}_{mapping_method[1]}_kmer_matches.csv"
            )
        )


@pytest.mark.parametrize(
    "kmer_matches, syn_topN, mapping_methods, error_msg",
    [
        (
            [
                {
                    "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                    "1": ["kmer_PC_" + str(x) for x in range(3)],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
            ],
            ["mapping_method"],
            "Mapping method not recognized",
        ),
        (
            [
                {
                    "0": ["kmer_AA_0", "kmer_AA_1", "kmer_AA_2"],
                    "1": ["kmer_PC_" + str(x) for x in range(3)],
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)] + ["kmer_AA_012"],
                },
            ],
            ["jurgen_schmidt"],
            "AA-kmers contain incorrect values.",
        ),
    ],
)
def test_match_kmers_error(tmpdir, kmer_matches, syn_topN, mapping_methods, error_msg):
    """test that the appropriate error is raised when function is provided incorrect mapping method"""
    # make and save a temporary file containing the matching kmer dataframe
    syn_topN_df = [pd.DataFrame(x) for x in syn_topN]
    with tmpdir.as_cwd():
        for i, kmer_matches_N in enumerate(kmer_matches):
            pd.DataFrame(kmer_matches_N).to_parquet(
                f"kmer_maps_{mapping_methods[i]}.parquet.gzip"
            )
    with pytest.raises(ValueError, match=error_msg):
        dtra_utils.match_kmers(syn_topN_df, mapping_methods, tmpdir)


def test_get_kmer_viruses():
    values = list(range(10))
    kmer_names = [f"kmer_PC_{n}" for n in values]
    kmer_map_names = [f"kmer_AA_{string.ascii_uppercase[n]}" for n in values]
    virus_names = [f"virus{n}" for n in values]
    protein_names = [f"protein{n}" for n in values]
    kmer_info = []
    for kmer, kmer_map, virus, protein in zip(
        kmer_names, kmer_map_names, virus_names, protein_names
    ):
        kmer_info.append(KmerData(None, kmer, kmer_map, virus, protein))

    virus_pairs_exp = {f"kmer_PC_{i}": [(f"virus{i}", f"protein{i}")] for i in values}
    kmer_info_df = pd.DataFrame([k.__dict__ for k in kmer_info])
    virus_pairs = dtra_utils.get_kmer_viruses(kmer_names, kmer_info_df)

    assert virus_pairs == virus_pairs_exp


@pytest.mark.parametrize(
    "accessions, kmer_type, mapping_method",
    [
        (["AC_000008.1", "NC_001563.2", "NC_039210.1"], "PC", "jurgen_schmidt"),
    ],
)
def test_save_load_all_kmer_info(tmpdir, accessions, kmer_type, mapping_method):
    kmer_info = []
    kmer_info_list = []
    test_records = []
    # make a dataframe that looks like all_kmer_info
    for accession in accessions:
        tests_dir = files("viral_seq") / "tests" / "cache_test" / accession
        test_records.append(_append_recs(tests_dir))

    _, kmer_info = get_kmers(
        test_records,
        kmer_type=kmer_type,
        mapping_method=mapping_method,
        gather_kmer_info=True,
    )
    kmer_info_list.extend(kmer_info)

    # recapitulate save_kmer_info dataframe handling
    all_kmer_info_df = pd.DataFrame([k.__dict__ for k in kmer_info_list])

    # save and load the file
    with tmpdir.as_cwd():
        dtra_utils.save_kmer_info(all_kmer_info_df, "all_kmer_info_test.parquet.gzip")
        kmer_info_load = dtra_utils.load_kmer_info("all_kmer_info_test.parquet.gzip")

    assert_frame_equal(all_kmer_info_df, kmer_info_load)


@pytest.mark.parametrize(
    "accessions, kmer_type, mapping_method",
    [
        (["AC_000008.1"], "PC", "jurgen_schmidt"),
    ],
)
def test_save_all_kmer_info(tmpdir, accessions, kmer_type, mapping_method):
    kmer_info = []
    all_kmer_info = []
    test_records = []
    # make a dataframe that looks like all_kmer_info
    for accession in accessions:
        tests_dir = files("viral_seq") / "tests" / "cache_test" / accession
        test_records.append(_append_recs(tests_dir))

    _, kmer_info = get_kmers(
        test_records,
        kmer_type=kmer_type,
        mapping_method=mapping_method,
        gather_kmer_info=True,
    )
    all_kmer_info.extend(kmer_info)
    all_kmer_info_df = pd.DataFrame([k.__dict__ for k in all_kmer_info])

    # save and load the file
    with tmpdir.as_cwd():
        dtra_utils.save_kmer_info(all_kmer_info_df, "all_kmer_info_test.parquet.gzip")
        assert os.path.exists("all_kmer_info_test.parquet.gzip")

    # get information from kmer_info for sanity check
    assert all_kmer_info_df.iloc[0].mapping_method == "jurgen_schmidt"
    assert all_kmer_info_df.iloc[0].protein_name == "E1A"
    assert all_kmer_info_df.iloc[0].virus_name == "Human adenovirus 5"
    assert all_kmer_info_df.iloc[0].kmer_names == "kmer_PC_6536613006"
    assert all_kmer_info_df.iloc[0].kmer_maps == "kmer_AA_MRHIICHGGV"
    assert all_kmer_info_df.iloc[0].virus_name == "Human adenovirus 5"
    assert all_kmer_info_df.shape == (11037, 5)


def test_load_all_kmer_info():
    load_file = files("viral_seq.tests.expected").joinpath(
        "all_kmer_info_test.parquet.gzip"
    )
    kmer_info_load_df = dtra_utils.load_kmer_info(load_file)

    # get information from kmer_info for sanity check
    assert kmer_info_load_df.iloc[0].mapping_method == "jurgen_schmidt"
    assert kmer_info_load_df.iloc[0].protein_name == "E1A"
    assert kmer_info_load_df.iloc[0].virus_name == "Human adenovirus 5"
    assert kmer_info_load_df.iloc[0].kmer_names == "kmer_PC_6536613006"
    assert kmer_info_load_df.iloc[0].kmer_maps == "kmer_AA_MRHIICHGGV"
    assert kmer_info_load_df.iloc[0].virus_name == "Human adenovirus 5"
    assert kmer_info_load_df.iloc[0].include_pair
    assert kmer_info_load_df.shape == (11037, 6)


def test_transform_kmer_data():
    # make a list of KmerData objects
    values = list(range(5))
    kmer_names = [f"kmer_PC_{v}" for v in values]
    kmer_maps = [f"kmer_AA_{string.ascii_uppercase[v]}" for v in values]
    viruses = [f"virus_{v}" for v in values]
    proteins = [f"proteins_{v}" for v in values]

    kmer_data_list = []
    for i in range(len(values)):
        kmer_data_list.append(
            KmerData(
                "test",
                [kmer_names[i]],
                [kmer_maps[i]],
                viruses[i],
                proteins[i],
            )
        )
    # expected dictionary
    df_exp = pd.DataFrame(
        {
            "mapping_method": {i: "test" for i in values},
            "kmer_names": {i: [f"kmer_PC_{i}"] for i in values},
            "kmer_maps": {i: [f"kmer_AA_{string.ascii_uppercase[i]}"] for i in values},
            "virus_name": {i: f"virus_{i}" for i in values},
            "protein_name": {i: f"proteins_{i}" for i in values},
        }
    )
    df_out = dtra_utils.transform_kmer_data(kmer_data_list)
    assert_frame_equal(df_out, df_exp)
