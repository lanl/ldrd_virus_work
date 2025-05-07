from viral_seq.analysis import dtra_utils
import pandas as pd
from numpy.testing import assert_array_equal
from importlib.resources import files
from viral_seq.analysis.spillover_predict import _append_recs
from viral_seq.analysis.get_features import get_kmers
from viral_seq import run_workflow as workflow
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


def test_get_kmer_viruses():
    kmer_names = [f"kmer_PC_{n}" for n in range(10)]
    virus_names = [f"virus{n}" for n in range(10)]
    protein_names = [f"protein{n}" for n in range(10)]
    kmer_info = []
    for kmer, virus, protein in zip(kmer_names, virus_names, protein_names):
        kmer_info.append(workflow.KmerData(None, [kmer], virus, protein))

    virus_pairs_exp = {
        "kmer_PC_0": [("virus0", "protein0")],
        "kmer_PC_1": [("virus1", "protein1")],
        "kmer_PC_2": [("virus2", "protein2")],
        "kmer_PC_3": [("virus3", "protein3")],
        "kmer_PC_4": [("virus4", "protein4")],
        "kmer_PC_5": [("virus5", "protein5")],
        "kmer_PC_6": [("virus6", "protein6")],
        "kmer_PC_7": [("virus7", "protein7")],
        "kmer_PC_8": [("virus8", "protein8")],
        "kmer_PC_9": [("virus9", "protein9")],
    }
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

    kmer_info, _ = get_kmers(
        test_records,
        kmer_type=kmer_type,
        mapping_method=mapping_method,
        kmer_info=kmer_info,
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

    kmer_info, _ = get_kmers(
        test_records,
        kmer_type=kmer_type,
        mapping_method=mapping_method,
        kmer_info=kmer_info,
    )
    all_kmer_info.extend(kmer_info)
    all_kmer_info_df = pd.DataFrame([k.__dict__ for k in all_kmer_info])

    # save and load the file
    with tmpdir.as_cwd():
        dtra_utils.save_kmer_info(all_kmer_info_df, "all_kmer_info_test.parquet.gzip")
        assert os.path.exists("all_kmer_info_test.parquet.gzip")

    # get information from kmer_info for sanity check
    assert all_kmer_info_df.iloc[0].protein_name == "E1A"
    assert all_kmer_info_df.iloc[0].virus_name == "Human adenovirus 5"
    assert all_kmer_info_df.iloc[0].kmer_names == ["kmer_PC_6536613006"]
    assert all_kmer_info_df.shape == (11037, 4)


def test_load_all_kmer_info():
    load_file = files("viral_seq.tests.expected").joinpath(
        "all_kmer_info_test.parquet.gzip"
    )
    kmer_info_load_df = dtra_utils.load_kmer_info(load_file)

    # get information from kmer_info for sanity check
    assert kmer_info_load_df.iloc[0].protein_name == "E1A"
    assert kmer_info_load_df.iloc[0].virus_name == "Human adenovirus 5"
    assert kmer_info_load_df.iloc[0].kmer_names == "kmer_PC_6536613006"
    assert kmer_info_load_df.shape == (11037, 4)


def test_transform_kmer_data():
    # make a list of KmerData objects
    values = list(range(5))
    kmer_names = ["kmer_" + str(v) for v in values]
    viruses = ["virus_" + str(v) for v in values]
    proteins = ["proteins_" + str(v) for v in values]

    kmer_data_list = []
    for i in range(len(values)):
        kmer_data_list.append(
            workflow.KmerData("test", [kmer_names[i]], viruses[i], proteins[i])
        )
    # expected dictionary
    df_exp = pd.DataFrame(
        {
            "mapping_method": {0: "test", 1: "test", 2: "test", 3: "test", 4: "test"},
            "kmer_names": {
                0: ["kmer_0"],
                1: ["kmer_1"],
                2: ["kmer_2"],
                3: ["kmer_3"],
                4: ["kmer_4"],
            },
            "virus_name": {
                0: "virus_0",
                1: "virus_1",
                2: "virus_2",
                3: "virus_3",
                4: "virus_4",
            },
            "protein_name": {
                0: "proteins_0",
                1: "proteins_1",
                2: "proteins_2",
                3: "proteins_3",
                4: "proteins_4",
            },
        }
    )
    df_out = dtra_utils.transform_kmer_data(kmer_data_list)
    assert_frame_equal(df_out, df_exp)
