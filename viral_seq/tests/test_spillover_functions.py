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
