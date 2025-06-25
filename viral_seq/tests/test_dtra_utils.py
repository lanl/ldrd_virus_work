from viral_seq.analysis import dtra_utils
import pandas as pd
from numpy.testing import assert_array_equal
from importlib.resources import files
import pytest
from pandas.testing import assert_frame_equal
import os


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


@pytest.mark.parametrize(
    "kmer_matches, syn_topN, kmer_matches_exp, mapping_methods",
    [
        (
            # this case tests that the function produces matches when two mapping methods are compared
            [
                {
                    0: {
                        "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y", "kmer_AA_G"],
                        "1": ["kmer_PC_" + str(x) for x in range(3)] + ["kmer_PC_0"],
                    },
                },
                {
                    0: {
                        "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_S"],
                        "1": ["kmer_PC_" + str(x + 1) for x in range(3)],
                    },
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
                "jurgen_schmidt": {0: "kmer_PC_0", 1: "kmer_PC_1"},
                "shen_2007": {0: "kmer_PC_1", 1: "kmer_PC_2"},
                "matching_AA_kmers": {
                    0: ["kmer_AA_G"],
                    1: ["kmer_AA_C"],
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # this test case checks that the correct values are returned when
        # the topN kmers contains kmer strings have len > 1
        (
            [
                {
                    0: {
                        "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                        "1": ["kmer_PC_" + str(x) for x in range(3)],
                    },
                    1: {
                        "0": ["kmer_AA_AG", "kmer_AA_GA"],
                        "1": ["kmer_PC_00"] * 2,
                    },
                },
                {
                    0: {
                        "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_S"],
                        "1": ["kmer_PC_" + str(x + 1) for x in range(3)],
                    },
                    1: {
                        "0": ["kmer_AA_AG", "kmer_AA_GA"],
                        "1": ["kmer_PC_11"] * 2,
                    },
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
                "matching_AA_kmers": {
                    0: ["kmer_AA_G"],
                    1: ["kmer_AA_AG", "kmer_AA_GA"],
                    2: ["kmer_AA_C"],
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # this test case checks that the correct matches are returned when
        # the PC kmers match between schemes, but they map to different AA kmers
        (
            [
                {
                    0: {
                        "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                        "1": ["kmer_PC_" + str(x) for x in range(3)],
                    },
                },
                {
                    0: {
                        "0": ["kmer_AA_Y", "kmer_AA_C", "kmer_AA_A"],
                        "1": ["kmer_PC_4", "kmer_PC_2", "kmer_PC_1"],
                    },
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
                "matching_AA_kmers": {
                    0: ["kmer_AA_C"],
                    1: ["kmer_AA_Y"],
                },
            },
            ["jurgen_schmidt", "shen_2007"],
        ),
        # this case tests that nothing is returned when only a single mapping method is used
        # to generate AA kmer matches.
        (
            [
                {
                    0: {
                        "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                        "1": ["kmer_PC_" + str(x) for x in range(3)],
                    },
                },
            ],
            [
                {
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
                },
            ],
            None,
            ["jurgen_schmidt"],
        ),
    ],
)
def test_match_kmers(tmpdir, kmer_matches, syn_topN, kmer_matches_exp, mapping_methods):
    """test that matching AA-kmers are found between the two mapping methods with different PC-kmers"""

    # make and save a temporary file containing the matching kmer dataframe
    syn_topN_df = [pd.DataFrame(x) for x in syn_topN]
    for i, kmer_matches_N in enumerate(kmer_matches):
        for k, kmer_matches_len in kmer_matches_N.items():
            pd.DataFrame(kmer_matches_len).to_parquet(
                f"{tmpdir}/kmer_maps_k{k+1}_{mapping_methods[i]}.parquet.gzip"
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
        assert os.path.exists(
            os.path.join(tmpdir, f"topN_PC_AA_kmer_mappings_{mm}.csv")
        )


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
                    "0": ["kmer_AA_G", "kmer_AA_C", "kmer_AA_Y"],
                    "1": ["kmer_PC_" + str(x) for x in range(3)],
                },
                {
                    "0": ["kmer_AA_F", "kmer_AA_M", "kmer_AA_H"],
                    "1": ["kmer_PC_" + str(x + 3) for x in range(3)],
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
                    "0": ["kmer_AA_F", "kmer_AA_C", "kmer_AA_K"],
                    "1": ["kmer_PC_7", "kmer_PC_1", "kmer_PC_5"],
                },
                {
                    "0": ["kmer_AA_F", "kmer_AA_C", "kmer_AA_K"],
                    "1": ["kmer_PC_4", "kmer_PC_2", "kmer_PC_7"],
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
            "Matching AA kmers between PC kmers:\n jurgen_schmidt shen_2007 matching_AA_kmers\n     kmer_PC_1 kmer_PC_2       [kmer_AA_C]\n     kmer_PC_5 kmer_PC_7       [kmer_AA_K]\n     kmer_PC_7 kmer_PC_4       [kmer_AA_F]",
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
                k = list(
                    set([len(s.replace("kmer_PC_", "")) for s in syn_topN[i]["0"]])
                )
                pd.DataFrame(kmer_matches[i]).to_parquet(
                    f"kmer_maps/kmer_maps_k{k[0]}_{mm}.parquet.gzip"
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
                    "0": ["kmer_PC_" + str(x) for x in range(3)],
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
    for i, kmer_matches_N in enumerate(kmer_matches):
        pd.DataFrame(kmer_matches_N).to_parquet(
            f"{tmpdir}/kmer_maps_k1_{mapping_methods[i]}.parquet.gzip"
        )
    with pytest.raises(ValueError, match=error_msg):
        dtra_utils.match_kmers(syn_topN_df, mapping_methods, tmpdir)
