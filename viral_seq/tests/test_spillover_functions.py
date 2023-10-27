from importlib.resources import files
import pytest
from click.testing import CliRunner
from viral_seq.cli.cli import cli
import json
import pandas as pd
from pandas.testing import assert_frame_equal

csv_train = files("viral_seq.tests").joinpath("TrainingSet.csv")
csv_test = files("viral_seq.tests").joinpath("TestSet.csv")
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
def test_network_cli_pull(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pull-data",
            "--email",
            email,
            "--cache",
            tmp_path.absolute().as_posix(),
            "--file",
            csv_train.absolute().as_posix(),
        ],
    )
    assert result.exit_code == 0
    assert (
        "number of records added to the local cache from online search: 14"
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
            cli, ["cross-validation", "--file", "table.parquet.gzip", "--splits", "2"]
        )
        assert result.exit_code == 0
        aucs = []
        for i in range(2):
            with open("cv_" + str(i) + "_metrics.json", "r") as f:
                data = json.load(f)
            aucs.append(data["AUC"])

        assert aucs == pytest.approx([0.8333333333333333, 0.875])
        # we can't check the image generated easily so we only verify the plot generation doesn't fail
        result = runner.invoke(
            cli, ["plot-roc", "cv_0_roc_curve.csv", "cv_1_roc_curve.csv"]
        )
        assert result.exit_code == 0
        result = runner.invoke(cli, ["train", "--file", "table.parquet.gzip"])
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
            cli, ["predict", "--file", "table.parquet.gzip", "--rfc-file", "rfc.p"]
        )
        assert result.exit_code == 0
        with open("cli_metrics.json", "r") as f:
            data = json.load(f)

        assert data["AUC"] == pytest.approx(0.38)


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


def test_expanded_kmers(tmp_path):
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


@pytest.mark.slow
def test_save_load_split_parquet(tmp_path):
    """Regression test the file splitting of large parquet files"""
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    csv_train_str = str(csv_train.resolve())
    str(files("viral_seq.tests").joinpath("test_expanded_kmers.csv").resolve())
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
                "-g",
                "-gc",
                "-kmers",
                "-k",
                "2 3 4 5 6 7 8 9 10",
                "-kmerspc",
                "-kpc",
                "2 3 4 5 6 7",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "train",
                "-f",
                "table.parquet.00.gzip table.parquet.01.gzip",
            ],
        )
        assert result.exit_code == 0
