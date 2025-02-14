from viral_seq.analysis import get_features
from viral_seq.analysis.get_features import get_genomic_features, get_kmers
from viral_seq.analysis.spillover_predict import _append_recs
import viral_seq.run_workflow as workflow
import pandas as pd
from pandas.testing import assert_frame_equal
import os
from importlib.resources import files
from Bio import SeqIO
import pytest
from collections import defaultdict
import numpy as np


@pytest.mark.parametrize(
    "accession, sep, file_name, calc_feats",
    [
        (
            "NC_019843.3",
            "\t",
            "MERS-CoV_features.csv",
            get_genomic_features,
        ),  # generic viral genomic feature test
        (
            "NC_007620.1",
            ",",
            "Menangle_features.csv",
            get_genomic_features,
        ),  # bad coding sequence test
        (
            "NC_007620.1",
            ",",
            "Menangle_features_kmers.csv",
            lambda e: get_kmers(e, k=2)[1],
        ),  # bad coding sequence kmer calculation test
        (
            "HM045787.1",
            ",",
            "Chikungunya_features.csv",
            get_genomic_features,
        ),  # ambiguous nucleotide test
    ],
)
@pytest.mark.filterwarnings("error:Partial codon")
def test_features(accession, sep, file_name, calc_feats):
    tests_dir = files("viral_seq") / "tests" / accession
    test_record = _append_recs(tests_dir)
    df = pd.DataFrame(calc_feats([test_record]), index=[accession]).reset_index()
    # For viral genomic features check our calculation matches published results in
    # https://doi.org/10.1371/journal.pbio.3001390
    # For k-mers regression test
    df_expected = pd.read_csv(files("viral_seq.tests").joinpath(file_name), sep=sep)
    assert_frame_equal(
        df.sort_index(axis=1),
        df_expected.sort_index(axis=1),
        check_names=True,
        rtol=1e-9,
        atol=1e-9,
    )


def test_genomic_features_bad_cds():
    # see: https://gitlab.lanl.gov/treddy/ldrd_virus_work/-/issues/15
    data = files("viral_seq.tests") / "cache_issue_15"
    data_path = os.path.join(data, "KU672593.1", "KU672593.1.genbank")
    records = [SeqIO.read(data_path, "genbank")]
    actual = get_genomic_features(records)
    assert len(actual) == 16


def test_aa_map_wrong_method():
    with pytest.raises(NotImplementedError, match="not supported"):
        get_features.aa_map("A", method="zzz")


def test_aa_map_wrong_input():
    with pytest.raises(ValueError, match="length 1"):
        get_features.aa_map("AC", method="shen_2007")


@pytest.mark.parametrize(
    "method, aa_in, aa_expected",
    [
        ("shen_2007", "A", "1"),
        ("shen_2007", "C", "2"),
        ("shen_2007", "P", "3"),
        ("shen_2007", "M", "4"),
        ("shen_2007", "Q", "5"),
        ("shen_2007", "E", "6"),
        ("shen_2007", "K", "7"),
        ("shen_2007", "Z", "*"),
        ("jurgen_schmidt", "A", "0"),
        ("jurgen_schmidt", "S", "2"),
        ("jurgen_schmidt", "F", "7"),
        ("jurgen_schmidt", "C", "1"),
        ("jurgen_schmidt", "P", "8"),
        ("jurgen_schmidt", "M", "6"),
        ("jurgen_schmidt", "Q", "3"),
        ("jurgen_schmidt", "E", "4"),
        ("jurgen_schmidt", "K", "5"),
        ("jurgen_schmidt", "J", "6"),
        ("jurgen_schmidt", "Z", "*"),
    ],
)
def test_aa_map(method, aa_in, aa_expected):
    actual = get_features.aa_map(aa_in, method=method)
    assert actual == aa_expected


def test_error_get_kmers():
    tests_dir = files("viral_seq") / "tests" / "NC_007620.1"
    test_record = _append_recs(tests_dir)
    with pytest.raises(ValueError, match="No mapping method required for AA-kmers."):
        get_kmers([test_record], kmer_type="AA", mapping_method="shen_2007")
    with pytest.raises(ValueError, match="Please specify mapping method for PC-kmers."):
        get_kmers([test_record], kmer_type="PC", mapping_method=None)


@pytest.mark.parametrize(
    "accession, kmer_type, mapping_method, exp_kmer, exp_len",
    [
        ("NC_007620.1", "AA", None, "kmer_AA_MSSVFRAFEL", 4489),
        ("NC_007620.1", "PC", "shen_2007", "kmer_PC_4441371363", 4489),
        ("NC_007620.1", "PC", "jurgen_schmidt", "kmer_PC_6226750746", 4489),
    ],
)
def test_get_kmers(accession, kmer_type, mapping_method, exp_kmer, exp_len):
    tests_dir = files("viral_seq") / "tests" / accession
    test_record = _append_recs(tests_dir)

    _, kmers = get_kmers(
        [test_record], kmer_type=kmer_type, mapping_method=mapping_method
    )
    mapped_kmers = list(kmers.keys())
    test_kmer = mapped_kmers[0]

    assert test_kmer == exp_kmer
    assert len(mapped_kmers) == exp_len


@pytest.mark.parametrize(
    "accessions, kmer_type, mapping_method",
    [
        (["AC_000008.1", "NC_001563.2", "NC_039210.1"], "PC", "jurgen_schmidt"),
    ],
)
def test_save_load_all_kmer_info(tmpdir, accessions, kmer_type, mapping_method):
    kmer_info = defaultdict(list)
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
    all_kmer_info.append(pd.DataFrame.from_dict(kmer_info, orient="index"))

    # save and load the file
    with tmpdir.as_cwd():
        workflow.save_kmer_info(all_kmer_info, "all_kmer_info_test.parquet.gzip")
        kmer_info_load = workflow.load_kmer_info("all_kmer_info_test.parquet.gzip")

    # recapitulate save_kmer_info dataframe handling
    all_kmer_info_df = pd.concat(all_kmer_info)
    all_kmer_info_df["Info"] = all_kmer_info_df.apply(
        lambda row: [x for x in row if x is not None and not pd.isna(x)], axis=1
    )
    all_kmer_info_df = all_kmer_info_df["Info"]
    all_kmer_info_df = all_kmer_info_df.to_frame()

    # because the object hashes are different between save and load dataframes
    # open up the objects and check the contents are the same for some random kmers
    # TODO: add assertion that checks if the kmer has > 1 entry and that they are not the same
    rng = np.random.default_rng(seed=2024)
    random_idx = rng.integers(0, len(kmer_info_load), 10)
    for idx in random_idx:
        save_protein = all_kmer_info_df.iloc[idx].Info[0].protein_name
        load_protein = kmer_info_load.iloc[idx].Info[0].protein_name
        save_virus = all_kmer_info_df.iloc[idx].Info[0].virus_name
        load_virus = kmer_info_load.iloc[idx].Info[0].virus_name
        save_kmer = all_kmer_info_df.iloc[idx].Info[0].kmer_names
        load_kmer = kmer_info_load.iloc[idx].Info[0].kmer_names
        assert save_protein == load_protein
        assert save_virus == load_virus
        assert save_kmer == load_kmer
