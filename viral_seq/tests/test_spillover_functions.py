from importlib.resources import files
import pytest
from click.testing import CliRunner
from viral_seq.cli.cli import cli
import json
import pandas as pd
from pandas.testing import assert_frame_equal
from viral_seq.analysis import spillover_predict as sp
import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

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


def test_load_multi_parquet(tmp_path):  # noqa: ARG001
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
    records = sp.load_from_cache(
        accessions=[accession], cache=cache_str, verbose=True, filter=False
    )
    assert len(records) == 1
    assert records[0].id == "HM147992.1"


def test_bounce_missing_accession():
    this_cache = files("viral_seq.tests") / "cache_unfiltered"
    cache_str = str(this_cache.resolve())
    with pytest.raises(ValueError, match="suitable entry"):
        sp.load_from_cache(
            accessions=["ABC1234.1"], cache=cache_str, verbose=True, filter=False
        )


@pytest.mark.parametrize("accession", ["HM147992", "HM147992.2"])
def test_build_table_bad_version(accession):
    this_cache = files("viral_seq.tests") / "cache_unfiltered"
    cache_str = str(this_cache.resolve())
    df = pd.DataFrame(
        [["1", accession, "Una virus"]], columns=["Unnamed: 0", "Accessions", "Species"]
    )
    table = sp.build_table(
        df=df, cache=cache_str, genomic=False, kmers=False, kmers_pc=False, gc=True
    )
    assert table["GC Content"].values[0] == pytest.approx(0.5050986292209964)


@pytest.mark.parametrize("target_column", ["Column1", "Column2"])
def test_build_table_target_column(target_column):
    this_cache = files("viral_seq.tests") / "cache"
    cache_str = str(this_cache.resolve())
    df = pd.read_csv(csv_train)
    df.rename({"Human Host": target_column}, inplace=True, axis=1)
    df_test = sp.build_table(
        df=df,
        cache=cache_str,
        genomic=False,
        kmers=True,
        kmer_k=[2],
        kmers_pc=False,
        gc=False,
        uni_select=True,
        num_select=10,
        target_column=target_column,
    )
    df_expected = pd.read_csv(
        files("viral_seq.tests") / "expected" / "target_column_test.csv"
    )
    df_test.drop(columns=[target_column], inplace=True)
    assert_frame_equal(
        df_test,
        df_expected,
        rtol=1e-9,
        atol=1e-9,
    )


def test_get_best_features():
    X, y = make_classification(
        n_samples=300,
        n_features=30,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=False,
        class_sep=2,
    )
    rfc = RandomForestClassifier(n_estimators=10, random_state=0)
    rfc.fit(X, y)
    feature_names = np.array([f"feature {i}" for i in range(30)])
    selected_features = np.sort(
        sp.get_best_features(rfc.feature_importances_, feature_names)
    )
    informative_features = feature_names[:3]
    npt.assert_array_equal(selected_features, informative_features)


@pytest.mark.parametrize(
    "size, percentile",
    [
        (10, 50),
        (10, 90),
        (100, 50),
        (100, 90),
        (1_000, 90),
    ],
)
def test_get_best_features_dirichlet(size, percentile):
    rng = np.random.default_rng(123)
    feature_importances = np.sort(rng.dirichlet(np.ones(size) * 200))
    feature_names = np.array([f"feature {i}" for i in range(size)])
    idx = int(percentile / 100.0 * size)
    selected_features = np.sort(
        sp.get_best_features(feature_importances, feature_names, percentile)
    )
    informative_features = feature_names[idx:]
    npt.assert_array_equal(selected_features, informative_features)


@pytest.mark.parametrize(
    "feature_importances, feature_names, percentile, match",
    [
        (
            np.array([1.0]),
            np.array(["feature"]),
            101,
            "percentile out of range",
        ),
        (np.array([1.0]), np.array(["feature"]), -1, "percentile out of range"),
        (
            np.array([0.5, 0.51]),
            np.array(["feature 1", "feature 2"]),
            90,
            "feature_importances must sum to 1",
        ),
        (
            np.array([1.0]),
            np.array(["feature 1", "feature 2"]),
            90,
            "feature_importances and feature_names must have the same shape",
        ),
    ],
)
def test_get_best_features_argument_guards(
    feature_importances, feature_names, percentile, match
):
    with pytest.raises(ValueError, match=match):
        sp.get_best_features(feature_importances, feature_names, percentile)


def test_issue_15():
    df = pd.DataFrame()
    df["Unnamed: 0"] = [0, 1]
    df["Species"] = ["No CDS Record", "Normal Record"]
    df["Accessions"] = ["KU672593.1", "HM119401.1"]
    cache_str = files("viral_seq.tests") / "cache_issue_15"
    df_expected = pd.read_csv(
        files("viral_seq.tests.expected") / "issue_15.csv", index_col=0
    )
    df_feats = sp.build_table(
        df,
        cache=cache_str,
        save=False,
        genomic=True,
        gc=False,
        kmers=False,
        kmers_pc=False,
    )
    # prior to fix this will only return a row for HM119401.1; post fix a row for each is returned
    assert_frame_equal(df_feats, df_expected)


def test_check_cache_tarball():
    for wf, tar_file in [
        ("DR", "dtra_cache.tar.gz"),
        ("DTRA", "cache_mollentze.tar.gz"),
    ]:
        with pytest.raises(ValueError, match="Extracted cache file"):
            sp.check_cache_tarball(wf, tar_file)
