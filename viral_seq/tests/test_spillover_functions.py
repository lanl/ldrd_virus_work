from importlib.resources import files
import pytest
from click.testing import CliRunner
from viral_seq.cli.cli import cli
import json

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
    runner = CliRunner()
    # we will test most/all of the modeling commands which use the output files of previous commands
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "calculate-table",
                "--cache",
                this_cache.absolute().as_posix(),
                "--file",
                csv_train.absolute().as_posix(),
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
            cli, ["cross-validation", "--file", "table.parquet.gzip", "--splits", "2"]
        )
        aucs = []
        for i in range(2):
            with open("cv_" + str(i) + "_metrics.json", "r") as f:
                data = json.load(f)
            aucs.append(data["AUC"])
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
                this_cache.absolute().as_posix(),
                "--file",
                csv_test.absolute().as_posix(),
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
        with open("cli_metrics.json", "r") as f:
            data = json.load(f)
        assert data["AUC"] == pytest.approx(0.3)
        assert result.exit_code == 0
