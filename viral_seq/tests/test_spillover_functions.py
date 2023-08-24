from importlib.resources import files
import pytest
from click.testing import CliRunner
from viral_seq.cli.cli import cli
import pandas as pd
from pathlib import Path

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
    assert (
        "number of records added to the local cache from online search: 1"
        in result.output
    )
    assert result.exit_code == 0


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
    assert (
        "number of records added to the local cache from online search: 14"
        in result.output
    )
    assert result.exit_code == 0


def test_verify_cache_cli():
    this_cache = files("viral_seq.tests") / "cache"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["verify-cache", "--cache", this_cache.absolute().as_posix()]
    )
    assert result.exit_code == 0


def test_modelling_cli():
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    csv_train_str = str(csv_train.resolve())
    csv_test_str = str(csv_test.resolve())
    runner = CliRunner()
    # we will test most/all of the modeling commands which use the output files of previous commands
    with runner.isolated_filesystem():
        print("Working dir:", Path.cwd())  # debug
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
        print(result.output)  # debug
        assert (
            "Saving the pandas DataFrame of genomic data to a parquet file"
            in result.output
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli, ["cross-validation", "--file", "table.parquet.gzip", "--splits", "2"]
        )
        print(result.output)  # debug
        print("Files in", Path.cwd())  # debug
        x = Path("./")  # debug
        print(list(filter(lambda y: y.is_file(), x.iterdir())))  # debug
        aucs = []
        for i in range(2):
            data = pd.read_csv("cv_" + str(i) + "_metrics.csv")
            aucs.append(data["AUC"].values[0])
        assert aucs == pytest.approx([0.5, 0.5])
        assert result.exit_code == 0
        # we can't check the image generated easily so we only verify the plot generation doesn't fail
        result = runner.invoke(
            cli, ["plot-roc", "cv_0_roc_curve.csv", "cv_1_roc_curve.csv"]
        )
        assert result.exit_code == 0
        result = runner.invoke(cli, ["train", "--file", "table.parquet.gzip"])
        assert "Saving random forest model to file" in result.output
        assert result.exit_code == 0
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
        assert (
            "Saving the pandas DataFrame of genomic data to a parquet file"
            in result.output
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli, ["predict", "--file", "table.parquet.gzip", "--rfc-file", "rfc.p"]
        )
        data = pd.read_csv("cli_metrics.csv")
        assert data["AUC"].values[0] == pytest.approx(0.3)
        assert result.exit_code == 0
