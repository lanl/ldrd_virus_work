from viral_seq.analysis import get_features
from viral_seq.analysis.get_features import (
    get_genomic_features,
    get_kmers,
    filter_structural_proteins,
    get_feature_idx,
)
from viral_seq.analysis.spillover_predict import _append_recs
import pandas as pd
from pandas.testing import assert_frame_equal
import os
from importlib.resources import files
from Bio import SeqIO
import pytest
from numpy.testing import assert_array_equal


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
            lambda e: get_kmers(e, k=2)[0],
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

    kmers, _ = get_kmers(
        [test_record], kmer_type=kmer_type, mapping_method=mapping_method
    )
    mapped_kmers = list(kmers.keys())
    test_kmer = mapped_kmers[0]

    assert test_kmer == exp_kmer
    assert len(mapped_kmers) == exp_len


@pytest.mark.parametrize(
    "filter_structural, exp_out",
    [
        # this case tests that the function finds the correct virus-protein
        # pairs with ``filter_structural=='surface_exposed'``
        (
            "surface_exposed",
            [False] * 6 + [True] + [False] * 10,
        ),
        # this case tests that the function finds the correct virus-protein
        # pairs with ``filter_structural=='not_surface_exposed'``
        (
            "not_surface_exposed",
            # NOTE: the presence of `False` values in the same positions
            # here and in the case above is the result of `polyprotein` entries
            # that get filtered out
            [False] + [True] * 5 + [False] + [True] * 8 + [False] + [True],
        ),
    ],
)
def test_filter_structural_proteins(filter_structural, exp_out):
    out = []
    tests_dir = (
        files("viral_seq.tests.cache_test") / "NC_001563.2"
    )  # aka West Nile Virus
    test_record = _append_recs(tests_dir)
    for feature in test_record.features:
        if feature.type in ["CDS", "mat_peptide"]:
            virus_name = test_record.annotations["organism"]
            protein_name = feature.qualifiers.get("product")[0]
            is_structural = filter_structural_proteins(
                virus_name, protein_name, filter_structural
            )
            out.append(is_structural)

    assert_array_equal(out, exp_out)


@pytest.mark.parametrize(
    "filter_structural, exp_len",
    [
        # test case for only surface exposed protein kmers
        ("surface_exposed", 488),
        # test case for only non-surface exposed protein kmers
        ("not_surface_exposed", 2893),
        # test case for gathering kmers from only ``CDS`` features
        (None, 3464),
        # test case for gathering kmers from ``CDS`` and ``mat_peptide`` features without ``polyproteins``
        ("all_features", 3381),
    ],
)
def test_get_kmers_filter_structural(filter_structural, exp_len):
    tests_dir = (
        files("viral_seq.tests.cache_test") / "NC_001563.2"
    )  # aka West Nile Virus
    test_record = _append_recs(tests_dir)

    kmers, _ = get_kmers(
        [test_record],
        kmer_type="PC",
        mapping_method="jurgen_schmidt",
        filter_structural=filter_structural,
    )

    assert len(kmers) == exp_len


@pytest.mark.parametrize(
    "accession, exp_out",
    [
        # this test case checks that the correct indices are returned by get_feature_idx
        # for the accession corresponding to ``West Nile Virus``
        ("NC_001563.2", (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20)),
        # this test case checks that the single polyprotein product idx is returned for ``FMDV``
        ("NC_039210.1", (2,)),
    ],
)
def test_get_feature_idx(accession, exp_out):
    tests_dir = files("viral_seq.tests.cache_test") / accession
    test_record = _append_recs(tests_dir)
    idx_out = get_feature_idx(test_record)

    assert idx_out == exp_out


def test_filter_structural_proteins_err():
    # this function only behaves properly with
    # its two expected values of `filter_structural` argument;
    # enforce proper error handling for other cases
    with pytest.raises(ValueError, match="filter_structural"):
        filter_structural_proteins(
            virus_name="influenza A",
            protein_name="neuraminidase",
            filter_structural="nonstandard_input",
        )
