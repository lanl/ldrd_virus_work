from importlib.resources import files
import pytest
from click.testing import CliRunner
from viral_seq.cli.cli import cli
import json
import pandas as pd
from pandas.testing import assert_frame_equal
from viral_seq.analysis import spillover_predict as sp

csv_train = files("viral_seq.tests").joinpath("TrainingSet.csv")
csv_test = files("viral_seq.tests").joinpath("TestSet.csv")
csv_partial = files("viral_seq.tests").joinpath("partial_record.csv")
email = "arhall@lanl.gov"


@pytest.mark.slow
def test_network_cli_search(tmp_path):
    runner = CliRunner()
    search_terms = "NC_045512.2"
    result = runner.invoke(
        cli,
        [
            "search-data",
            "--email",
            email,
            "--cache",
            tmp_path.absolute().as_posix(),
            "--query",
            search_terms,
            "--retmax",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert (
        "number of records added to the local cache from online search: 1"
        in result.output
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "file, no_filter, num_expected",
    [
        (csv_train, False, 14),
        (csv_partial, False, 0),
        (csv_partial, True, 1),
    ],
)
def test_network_cli_pull(tmp_path, file, no_filter, num_expected):
    runner = CliRunner()
    args = [
        "pull-data",
        "--email",
        email,
        "--cache",
        tmp_path.absolute().as_posix(),
        "--file",
        file.absolute().as_posix(),
    ]
    if no_filter:
        args += ["--no-filter"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert (
        "number of records added to the local cache from online search: "
        + str(num_expected)
        in result.output
    )


def test_verify_cache_cli():
    this_cache = files("viral_seq.tests") / "cache"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["verify-cache", "--cache", this_cache.absolute().as_posix()]
    )
    assert result.exit_code == 0


def test_modelling_cli():
    # regression testing modelling related cli commands
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    csv_train_str = str(csv_train.resolve())
    csv_test_str = str(csv_test.resolve())
    runner = CliRunner()
    # we will test most/all of the modeling commands which use the output files of previous commands
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_train_str,
                "-g",
                "-gc",
                "-kmers",
                "-k",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert (
            "Saving the pandas DataFrame of genomic data to a parquet file"
            in result.output
        )
        result = runner.invoke(
            cli, ["cross-validation", "table.parquet.gzip", "--splits", "2"]
        )
        assert result.exit_code == 0
        aucs = []
        for i in range(2):
            with open("cv_" + str(i) + "_metrics.json", "r") as f:
                data = json.load(f)
            aucs.append(data["AUC"])

        assert aucs == pytest.approx([0.9166666666666667, 0.8333333333333333])
        # we can't check the image generated easily so we only verify the plot generation doesn't fail
        result = runner.invoke(
            cli, ["plot-roc", "cv_0_roc_curve.csv", "cv_1_roc_curve.csv"]
        )
        assert result.exit_code == 0
        result = runner.invoke(cli, ["train", "table.parquet.gzip"])
        assert result.exit_code == 0
        assert "Saving random forest model to file" in result.output
        # table caclulation of a test set, which utilizes a trained random forest model
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_test_str,
                "--rfc-file",
                "rfc.p",
                "-g",
                "-gc",
                "-kmers",
                "-k",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert (
            "Saving the pandas DataFrame of genomic data to a parquet file"
            in result.output
        )
        result = runner.invoke(
            cli, ["predict", "table.parquet.gzip", "--rfc-file", "rfc.p"]
        )
        assert result.exit_code == 0
        with open("cli_metrics.json", "r") as f:
            data = json.load(f)

        assert data["AUC"] == pytest.approx(0.4)


@pytest.mark.slow
def test_human_similarity_features(tmp_path):
    # regression test calculating similarity features
    data_file = str(
        files("viral_seq.data").joinpath("ISG_transcript_ids.txt").resolve()
    )
    expected_table = str(
        files("viral_seq.tests").joinpath("test_similarity.csv").resolve()
    )
    search_file = tmp_path / "ids.txt"
    with open(data_file) as f:
        with open(search_file, "w") as o:
            o.writelines(" ".join(f.readlines()[0].split()[:100]))
    csv_train_str = str(csv_train.resolve())
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "pull-ensembl-transcripts",
                "--file",
                search_file,
                "--cache",
                tmp_path.absolute().as_posix(),
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_train_str,
                "-o",
                "table.parquet.gzip",
                "-g",
                "-sg",
                "-sc",
                tmp_path.absolute().as_posix(),
            ],
        )
        assert result.exit_code == 0
        df_test = pd.read_parquet("table.parquet.gzip")
        df_expected = pd.read_csv(expected_table)
        assert_frame_equal(
            df_test,
            df_expected,
            rtol=1e-9,
            atol=1e-9,
        )


def test_expanded_kmers():
    # regression test calculation of kmers, including PC kmers and calculating multiple kmers at once
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    csv_train_str = str(csv_train.resolve())
    expected_table = str(
        files("viral_seq.tests").joinpath("test_expanded_kmers.csv").resolve()
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_train_str,
                "-kmers",
                "-k",
                "2 3",
                "-kmerspc",
                "-kpc",
                "3 4",
            ],
        )
        assert result.exit_code == 0
        assert (
            "Saving the pandas DataFrame of genomic data to a parquet file"
            in result.output
        )
        df_test = pd.read_parquet("table.parquet.gzip")
        df_expected = pd.read_csv(expected_table)
        assert_frame_equal(
            df_test,
            df_expected,
            rtol=1e-9,
            atol=1e-9,
        )


@pytest.mark.parametrize("uni_type", ["chi2", "mutual_info_classif"])
def test_univariate_selection(uni_type):
    # regression test for feature selection
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    csv_train_str = str(csv_train.resolve())
    expected_table = str(
        files("viral_seq.tests").joinpath(f"test_uni_select_{uni_type}.csv").resolve()
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_train_str,
                "-kmers",
                "-k",
                "2",
                "-u",
                "-ut",
                f"{uni_type}",
                "-n",
                "100",
            ],
        )
        assert result.exit_code == 0
        assert (
            "Saving the pandas DataFrame of genomic data to a parquet file"
            in result.output
        )
        df_test = pd.read_parquet("table.parquet.gzip")
        df_expected = pd.read_csv(expected_table)
        assert_frame_equal(
            df_test,
            df_expected,
            rtol=1e-9,
            atol=1e-9,
        )


def test_load_multi_parquet(tmp_path):
    """Regression test joining parquet files created with multiple calculate-table commands"""
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    csv_train_str = str(csv_train.resolve())
    csv_test_str = str(csv_test.resolve())
    runner = CliRunner()
    with runner.isolated_filesystem():
        # test joining of the output of calculate-table commands is equal to output of calculating the same thing in one command
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_train_str,
                "-kmers",
                "-k",
                "2",
                "-o",
                "table.k2.parquet.gzip",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_train_str,
                "-kmerspc",
                "-kpc",
                "2",
                "-o",
                "table.kpc2.parquet.gzip",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_train_str,
                "-kmers",
                "-k",
                "2",
                "-kmerspc",
                "-kpc",
                "2",
                "-o",
                "table.expected.parquet.gzip",
            ],
        )
        assert result.exit_code == 0
        df_test = sp.load_files(("table.k2.parquet.gzip", "table.kpc2.parquet.gzip"))
        df_expected = pd.read_parquet("table.expected.parquet.gzip")
        assert_frame_equal(
            df_test,
            df_expected,
            rtol=1e-9,
            atol=1e-9,
        )
        # test cross-validation accepts multiple tables
        result = runner.invoke(
            cli,
            [
                "cross-validation",
                "table.k2.parquet.gzip",
                "table.kpc2.parquet.gzip",
                "--splits",
                "2",
            ],
        )
        assert result.exit_code == 0
        # regression test on cross-validation output
        aucs = []
        for i in range(2):
            with open("cv_" + str(i) + "_metrics.json", "r") as f:
                data = json.load(f)
            aucs.append(data["AUC"])

        assert aucs == pytest.approx([1.0, 0.6666666666666666])

        # test data to use for predict command
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_test_str,
                "-kmers",
                "-k",
                "2",
                "-o",
                "test_table.k2.parquet.gzip",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                cache_str,
                "--file",
                csv_test_str,
                "-kmerspc",
                "-kpc",
                "2",
                "-o",
                "test_table.kpc2.parquet.gzip",
            ],
        )
        assert result.exit_code == 0
        # check training accepts two tables
        result = runner.invoke(
            cli,
            [
                "train",
                "table.k2.parquet.gzip",
                "table.kpc2.parquet.gzip",
                "-o",
                "rfc.p",
            ],
        )
        assert result.exit_code == 0
        # check predict accepts two tables
        result = runner.invoke(
            cli,
            [
                "predict",
                "test_table.k2.parquet.gzip",
                "test_table.kpc2.parquet.gzip",
                "--rfc-file",
                "rfc.p",
            ],
        )
        assert result.exit_code == 0
        with open("cli_metrics.json", "r") as f:
            data = json.load(f)
        # regression test of predict output
        assert data["AUC"] == pytest.approx(0.38)


def test_build_table_with_partial():
    """check we can calculate a feature table with an accession labeled 'partial'"""
    this_cache = files("viral_seq.tests") / "cache_unfiltered"
    cache_str = str(this_cache.resolve())
    csv_partial_str = str(csv_partial.resolve())
    df = pd.read_csv(csv_partial_str)
    sp.build_table(df, cache=cache_str, kmers=True, kmer_k=[2])


@pytest.mark.parametrize("accession", ["HM147992", "HM147992.2"])
def test_load_accession_bad_version(accession):
    # check we can load accessions missing version information or with version information that doesn't match the cached accession
    this_cache = files("viral_seq.tests") / "cache_unfiltered"
    cache_str = str(this_cache.resolve())
    sp.load_from_cache(
        accessions=[accession], cache=cache_str, verbose=True, filter=False
    )
