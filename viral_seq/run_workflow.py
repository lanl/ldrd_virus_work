from importlib.resources import files
from viral_seq.analysis import spillover_predict as sp
from viral_seq.analysis import rfc_utils, classifier, feature_importance
from viral_seq.cli import cli
import pandas as pd
import re
import argparse
from http.client import IncompleteRead
from urllib.error import URLError
import polars as pl
import ast
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path
from warnings import warn
import json
from typing import Dict, Any
import matplotlib
import matplotlib.pyplot as plt
from time import perf_counter
import tarfile
import shap

matplotlib.use("Agg")


def validate_feature_table(file_name, idx, prefix):
    print("Validating", file_name)
    df = pl.read_parquet(file_name).to_pandas()
    expected_shape = table_info.iloc[idx][prefix + "_shape"]
    if not np.array_equal(df.shape, expected_shape):
        raise ValueError(
            "Feature table shape does not match what was precalculated.\nActual: %s\nExpected: %s"
            % (df.shape, expected_shape)
        )
    actual_sum = df.select_dtypes(["number"]).to_numpy().sum()
    expected_sum = table_info.iloc[idx][prefix + "_sum"]
    if not np.allclose(actual_sum, expected_sum, rtol=1e-8):
        raise ValueError(
            "The feature table's numerical sum, which sums all feature values for all viruses, does not match what was precalculated. Verify integrity of input data and feature calculation.\nActual: %s\nExpected: %s"
            % (actual_sum, expected_sum),
        )


def build_cache(cache_checkpoint=3, debug=False):
    """Download and store all data needed for the workflow"""
    if cache_checkpoint == "extract":
        with tarfile.open(cache_file, "r") as tar:
            tar.extractall(cache_extract_path)
        with open(extract_cookie, "w") as f:
            pass
        cache_checkpoint = 0
    cache_checkpoint = int(cache_checkpoint)
    if cache_checkpoint > 0:
        extract_cookie.unlink(missing_ok=True)
        print("Will pull down data to local cache")

    if debug:
        if extract_cookie.is_file():
            print(
                "Debug mode: cache extracted from file. Assertions on cache are unnecessary as it is expected files in cache may have diverged from those on NCBI. To generate cache using live data use option '--cache 3'"
            )
            debug = False
        else:
            print("Debug mode: will run assertions on generated cache")

    email = "arhall@lanl.gov"

    if cache_checkpoint > 2:
        print("Pulling viral sequence data to local cache...")
        try:
            for file in viral_files:
                cli.pull_data(
                    [
                        "--email",
                        email,
                        "--cache",
                        cache_viral,
                        "--file",
                        file,
                        "--no-filter",
                    ],
                    standalone_mode=False,
                )
        except (IncompleteRead, URLError):
            raise ConnectionError(
                "Connection closed as protocol synchronisation is probably lost. Re-run workflow with option --cache 3 and an active internet connection."
            ) from None

    if debug:
        print("Validating train and test cache...")
        df = pd.read_csv(viral_files[0])
        accessions_train = set([s for s in (" ".join(df["Accessions"].values)).split()])
        df = pd.read_csv(viral_files[1])
        accessions_test = set([s for s in (" ".join(df["Accessions"].values)).split()])
        cache_path, cache_subdirectories = sp.init_cache(cache_viral)
        cached_accessions = set(s.parts[-1] for s in cache_subdirectories)
        cached_accessions_stems = set(s.split(".")[0] for s in cached_accessions)
        cached_train = cached_accessions.intersection(accessions_train)
        cached_test = cached_accessions.intersection(accessions_test)
        # assert what's missing just differs by version with something present
        for this_set, that_set in zip(
            [cached_train, cached_test], [accessions_train, accessions_test]
        ):
            missing = that_set.difference(this_set)
            if len(missing) > 0:
                for each in missing:
                    this_stem = each.split(".")[0]
                    assert this_stem in cached_accessions_stems
                    print(
                        each,
                        "not in cache, but suitable accession present:",
                        [s for s in cached_accessions if s.startswith(this_stem)][0],
                    )
        num_dir_expected = len(accessions_train.union(accessions_test))
        num_dir_actual = len(cache_subdirectories)
        assert num_dir_expected == num_dir_actual, (
            "The number of directories in the cache differs from the number of accessions searched. There should only be one directory per accession.\nNumber of directories in cache: %s\nNumber of accessions searched: %s"
            % (num_dir_actual, num_dir_expected)
        )

    housekeeping_Trav = data.joinpath("Housekeeping_accessions.txt")
    file = housekeeping_Trav.absolute().as_posix()

    if cache_checkpoint > 1:
        print(
            "Pulling sequence data of human housekeeping genes for similarity features..."
        )
        with open(file, "r") as f:
            transcripts = f.readlines()[0].split()
        search_term = "(biomol_mrna[PROP] AND refseq[filter]) AND("
        first = True
        for transcript in transcripts:
            if first:
                first = False
            else:
                search_term += "OR "
            search_term += transcript + "[Accession] "
        search_term += ")"
        results = sp.run_search(
            search_terms=[search_term], retmax=len(transcripts), email=email
        )
        try:
            records = sp.load_results(results, email=email)
        except (IncompleteRead, URLError):
            raise ConnectionError(
                "Connection closed as protocol synchronisation is probably lost. Re-run workflow with option --cache 2 and an active internet connection."
            ) from None

        sp.add_to_cache(records, just_warn=True, cache=cache_hk)
    if debug:
        print("Validating human housekeeping gene cache...")
        with open(file, "r") as f:
            accessions_hk = set([s.split(".")[0] for s in (f.readlines()[0]).split()])
        cache_path, cache_subdirectories = sp.init_cache(cache_hk)
        cached_accessions = set(s.parts[-1].split(".")[0] for s in cache_subdirectories)
        missing_hk_file = data / "missing_hk.txt"
        cached_hk = cached_accessions.intersection(accessions_hk)
        missing_hk = accessions_hk.difference(cached_hk)
        extra_accessions = cached_accessions.difference(accessions_hk)
        if len(missing_hk) > 0:
            renamed_accessions = {}
            # some accessions have been replaced
            print("Checking if missing accessions have been renamed...")
            for accession in missing_hk:
                search_term = (
                    "(biomol_mrna[PROP] AND refseq[filter]) AND "
                    + accession
                    + "[Accession]"
                )
                results = sp.run_search(
                    search_terms=[search_term], retmax=1, email=email
                )
                if len(results) > 0:
                    assert len(results) == 1, (
                        "Expected search on accession %s to return only one replacement but received multiple results: %s"
                        % (accession, results)
                    )
                    new_name = list(results)[0].split(".")[0]
                    renamed_accessions[accession] = new_name
            for k, v in renamed_accessions.items():
                if k != v:
                    print(k, "was renamed as", v)
                    extra_accessions.remove(v)
                    missing_hk.remove(k)
                else:
                    raise ValueError(
                        "The accession "
                        + k
                        + " can currently be found on NCBI but is not in the cache. Rebuild the cache with option --cache 2",
                    )
            assert len(extra_accessions) == 0, (
                "There are accessions in the cache beyond what was searched. These accessions should be removed: %s"
                % extra_accessions
            )
        if len(missing_hk) > 0:  # len could have changed
            # currently we don't expect to find everything
            print(
                "Couldn't find",
                len(missing_hk),
                "accessions for housekeeping genes; saved to",
                missing_hk_file.absolute().as_posix(),
            )
            with open(missing_hk_file, "w") as f:
                print("Missing accessions for human housekeeping genes", file=f)
                print(missing_hk, file=f)
        # TODO: stricter assertion when we expect to find all accessions
        assert len(cached_accessions) + len(missing_hk) == len(accessions_hk), (
            "The number of missing accessions (%s) plus the number of cached accessions (%s) doesn't match the number of accessions searched (%s). This is likely due to an error in how these values were counted."
            % (len(missing_hk), len(cached_accessions), len(accessions_hk))
        )

    isg_Trav = data.joinpath("ISG_transcript_ids.txt")
    file = isg_Trav.absolute().as_posix()
    if cache_checkpoint > 0:
        try:
            print("Pulling sequence data of human isg genes for similarity features...")
            cli.pull_ensembl_transcripts(
                [
                    "--email",
                    email,
                    "--cache",
                    cache_isg,
                    "--file",
                    file,
                ],
                standalone_mode=False,
            )
        except (IncompleteRead, URLError):
            raise ConnectionError(
                "Connection closed as protocol synchronisation is probably lost. Re-run workflow with option --cache 1 and an active internet connection."
            ) from None

    if debug:
        print("Validating human ISG gene cache...")
        with open(file, "r") as f:
            ensembl_ids = set((f.readlines()[0]).split())
        cache_path, cache_subdirectories = sp.init_cache(cache_isg)
        missing = []
        missing_file = data / "missing_isg.txt"
        # see if we can find the ensembl transcript ids in the raw text of the genbank files
        cache_subdir_set = set()
        prog = re.compile(r"MANE Ensembl match\s+:: (ENST\d+)")
        # make a set of ensembl trascripts present
        for subdir in cache_subdirectories:
            genbank_file = list(subdir.glob("*.genbank"))[0]
            with open(genbank_file, "r") as f:
                match = prog.search(f.read())
                if match is not None:
                    cache_subdir_set.add(match.group(1))
        assert len(cache_subdir_set) == len(cache_subdirectories), (
            "Number of sequences linked to an Ensembl transcript does not match number of sequences in the cache. This is likely due to preexisting data in the cache. Clear the cache and try again.\nActual matches: %s\nExpected matches: %s"
            % (len(cache_subdir_set), len(cache_subdirectories))
        )
        missing = ensembl_ids.difference(cache_subdir_set)
        assert len(missing) + len(cache_subdir_set) == len(ensembl_ids), (
            "The number of missing Ensembl transcripts (%s) plus the number of cached transcripts (%s) doesn't match the number of transcripts searched (%s). This is likely due to an error in how these values were counted."
            % (len(missing), len(cache_subdir_set), len(ensembl_ids))
        )
        if len(missing) > 0:
            print(
                "Couldn't find",
                len(missing),
                "transcripts; ids saved to",
                missing_file.absolute().as_posix(),
            )
            with open(missing_file, "w") as f:
                print("Missing ISG transcripts", file=f)
                print(missing, file=f)


def build_tables(feature_checkpoint=0, debug=False):
    """Calculate all features and store in data tables for future use in the workflow"""

    if feature_checkpoint > 0:
        print("Will build feature tables for training models")

    if debug:
        if extract_cookie.is_file():
            print("Debug mode: will run assertions on generated tables")
        else:
            print(
                "Debug mode: cache not extracted from file, table assertions cannot be enforced"
            )
            debug = False

    if workflow == "DR":
        # tables are built in multiple parts as this is faster for reading/writing
        for i, (file, folder) in enumerate(zip(viral_files, table_locs)):
            prefix = "Train" if i == 0 else "Test"
            this_checkpoint_modifier = (1 - i) * 8
            this_checkpoint = 8 + this_checkpoint_modifier
            this_outfile = folder + "/" + prefix + "_main.parquet.gzip"
            if feature_checkpoint >= this_checkpoint:
                print(
                    "Building table for",
                    prefix,
                    "which includes genomic features, gc content, kmers with k=2,3,4, pc kmers with k=2,3,4,5,6, and similarity features for ISG and housekeeping genes.",
                )
                print("To restart at this point use --features", this_checkpoint)
                cli.calculate_table(
                    [
                        "--file",
                        file,
                        "--cache",
                        cache_viral,
                        "--outfile",
                        this_outfile,
                        "--features-genomic",
                        "--features-gc",
                        "--features-kmers",
                        "--kmer-k",
                        "2 3 4",
                        "--features-kmers-pc",
                        "--kmer-k-pc",
                        "2 3 4 5 6",
                        "--similarity-genomic",
                        "--similarity-cache",
                        cache_isg + " " + cache_hk,
                        "--target-column",
                        target_column,
                    ],
                    standalone_mode=False,
                )
            if debug:
                idx = np.abs(8 - (this_checkpoint - this_checkpoint_modifier))
                validate_feature_table(this_outfile, idx, prefix)
            this_checkpoint -= 1
            this_outfile = folder + "/" + prefix + "_kpc7.parquet.gzip"
            if feature_checkpoint >= this_checkpoint:
                print("Building table for", prefix, "which includes pc kmers with k=7.")
                print("To restart at this point use --features", this_checkpoint)
                cli.calculate_table(
                    [
                        "--file",
                        file,
                        "--cache",
                        cache_viral,
                        "--outfile",
                        this_outfile,
                        "--features-kmers-pc",
                        "--kmer-k-pc",
                        "7",
                        "--target-column",
                        target_column,
                    ],
                    standalone_mode=False,
                )
            if debug:
                idx = np.abs(8 - (this_checkpoint - this_checkpoint_modifier))
                validate_feature_table(this_outfile, idx, prefix)
            for k in range(5, 11):
                this_checkpoint -= 1
                this_outfile = folder + "/" + prefix + "_k{}.parquet.gzip".format(k)
                if feature_checkpoint >= this_checkpoint:
                    print(
                        "Building table for",
                        prefix,
                        "which includes kmers with k={}.".format(k),
                    )
                    print("To restart at this point use --features", this_checkpoint)
                    cli.calculate_table(
                        [
                            "--file",
                            file,
                            "--cache",
                            cache_viral,
                            "--outfile",
                            this_outfile,
                            "--features-kmers",
                            "--kmer-k",
                            str(k),
                            "--target-column",
                            target_column,
                        ],
                        standalone_mode=False,
                    )
                if debug:
                    idx = np.abs(8 - (this_checkpoint - this_checkpoint_modifier))
                    validate_feature_table(this_outfile, idx, prefix)

    elif workflow == "DTRA":
        for i, (file, folder) in enumerate(zip(viral_files, table_locs)):
            if i == 0:
                prefix = "Train"
                debug = False
                this_outfile = folder + "/" + prefix + "_main.parquet.gzip"
                feature_checkpoint = 10
                this_checkpoint = feature_checkpoint

                for k in range(6, 11):
                    this_checkpoint -= 2
                    this_outfile = folder + "/" + prefix + "_k{}.parquet.gzip".format(k)
                    if feature_checkpoint >= this_checkpoint:
                        print(
                            "Building table for Train",
                            "which includes kmers and pc kmers with k={}.".format(k),
                        )
                        print(
                            "To restart at this point use --features", this_checkpoint
                        )
                        cli.calculate_table(
                            [
                                "--file",
                                file,
                                "--cache",
                                cache_viral,
                                "--outfile",
                                this_outfile,
                                "--features-kmers",
                                "--kmer-k",
                                str(k),
                                "--features-kmers-pc",
                                "--kmer-k-pc",
                                str(k),
                                "--target-column",
                                target_column,
                            ],
                            standalone_mode=False,
                        )


def feature_selection_rfc(feature_selection, debug, n_jobs, random_state):
    """Sub-select features using best performing from a trained random forest classifier"""
    if feature_selection == "yes" or feature_selection == "none":
        print("Loading all feature tables for train...")
        train_files = tuple(glob(table_loc_train + "/*gzip"))
        X, y = sp.get_training_columns(
            table_filename=train_files, class_column=target_column
        )
        if feature_selection == "none":
            print(
                "All training features will be used as X_train in the following steps."
            )
        elif feature_selection == "yes":
            print(
                "Will train a random forest classifier to select the best performing features to use as X_train."
            )
            rfc = RandomForestClassifier(
                n_estimators=2_000,
                random_state=random_state,
                n_jobs=n_jobs,
                max_depth=30,
            )
            rfc.fit(X, y)
            if debug:
                print("Debug mode: Checking OOB score of Random Forest.")
                oob_score = rfc_utils.oob_score(
                    rfc, X, y, roc_auc_score, n_jobs=n_jobs, scoring_on_pred=False
                )
                print("Achieved an OOB score (AUC) of", oob_score)
                assert oob_score > 0.75, (
                    "OOB score of RandomForestClassifier is too poor for feature ranking to be reliable.\nActual score: %s\nExpected score: > 0.75"
                    % oob_score
                )
            keep_feats_rf = sp.get_best_features(
                rfc.feature_importances_, rfc.feature_names_in_
            )
            print("Selected", len(keep_feats_rf), "features from native RF")
            explainer = shap.Explainer(rfc, seed=random_state)
            shap_values = explainer(X)
            positive_shap_values = feature_importance.get_positive_shap_values(
                shap_values
            )
            mean_abs_shap_values = positive_shap_values.abs.mean(0).values
            mean_abs_shap_values /= mean_abs_shap_values.sum()
            keep_feats_shap = sp.get_best_features(mean_abs_shap_values, X.columns)
            print("Selected", len(keep_feats_shap), "features from RF SHAP")
            keep_feats = np.intersect1d(keep_feats_rf, keep_feats_shap)
            print(
                "Keeping",
                len(keep_feats),
                "features from intersection of the RF and RF SHAP approaches",
            )
            if debug:
                assert len(keep_feats) < 50_000, (
                    "Too many features selected by feature selection.\nAcutal: %s\nExpected: < 50_000"
                    % len(keep_feats)
                )
            X = X[keep_feats]
            print("Saving X_train to", table_loc_train_best)
            X.to_parquet(table_loc_train_best)
    elif feature_selection == "skip":
        print("Will use previously calculated X_train stored at", table_loc_train_best)
        X = pl.read_parquet(table_loc_train_best).to_pandas()
        y = pd.read_csv(train_file)[target_column]
    if debug and extract_cookie.is_file():
        # these might not exist if the workflow has only been run with --feature-selection none
        if Path(table_loc_train_best).is_file():
            validate_feature_table(table_loc_train_best, 8, "Train")
        else:
            warn(
                "File at {} cannot be validated because it does not exist. If using option '--feature-selection none' this is expected behavior.".format(
                    table_loc_train_best
                ),
                UserWarning,
            )
    return X, y


def optimize_model(
    model,
    X_train,
    y_train,
    outfile,
    config,
    num_samples,
    optimize="skip",
    name="Classifier",
    debug=False,
    random_state=123,
    n_jobs_cv=1,
    n_jobs=1,  # only used for default score
):
    if optimize == "yes":
        print(
            "Performing hyperparameter optimization with target AUC across 5 fold Cross Validation"
        )
        res = classifier.get_hyperparameters(
            model=model,
            config=config,
            num_samples=num_samples,
            X=X_train,
            y=y_train,
            n_jobs_cv=n_jobs_cv,
            random_state=random_state,
        )
        print("Checking default score...")
        default_score = classifier.cv_score(
            model,
            X=X_train,
            y=y_train,
            random_state=random_state,
            n_estimators=2_000,
            n_jobs=n_jobs,
        )
        print("ROC AUC with default settings:", default_score)
        res["targets"] = [default_score] + res["targets"]
        if default_score > res["target"]:
            print(
                "Will use default settings since we weren't able to improve with optimization."
            )
            res["target"] = default_score
            res["params"] = {}
        print("Saving results to", outfile)
        with open(outfile, "w") as f:
            json.dump(res, f)
    elif optimize == "skip":
        print("Loading previously saved parameters from", outfile)
        with open(outfile, "r") as f:
            res = json.load(f)
    if debug:
        print(
            "Debug mode: asserting best score during hyperparameter search is sufficient"
        )
        assert res["target"] > 0.75, (
            "ROC AUC achieved is too poor to accept hyperparameter optimization.\nActual score: %s\nExpected score: > 0.75"
            % res["target"]
        )

    print("Hyperparameters that performed best:", res["params"])
    return res


def optimization_plots(input_data: Dict[str, Any], out_source: str, out_fig: str):
    for name, targets in input_data.items():
        target_max = np.maximum.accumulate(targets)
        df = pd.DataFrame(target_max, columns=[name])
        if Path(out_source).is_file():
            old_df = pd.read_csv(out_source)
            df = pd.concat([old_df.loc[:, list(old_df.columns != name)], df], axis=1)
        print("Writing optimization plot data to", out_source)
        df.to_csv(out_source, index=False)
    print(
        "Generating plot of optimization progress for classifiers and saving to",
        out_fig,
    )
    fig, ax = plt.subplots(1, 1)
    df.plot(ax=ax, alpha=0.6, marker="o")
    ax.set_title("Maximum Optimization Target\nAUC over 5 folds")
    ax.set_xlabel("Step")
    ax.set_ylabel("AUC")
    ax.legend(loc=4, fontsize=6)
    fig.savefig(out_fig, dpi=300)


def get_test_features(
    table_loc_test,
    table_loc_test_saved,
    test_file,
    X_train,
    extract_cookie,
    debug=False,
):
    if not extract_cookie.is_file():
        debug = False
    print("Ensuring X_test has only the features in X_train...")
    if Path(table_loc_test_saved).exists():
        X_test = pl.read_parquet(table_loc_test_saved).to_pandas()
        y_test = pd.read_csv(test_file)["Human Host"]
        if set(X_test.columns) == set(X_train.columns):
            print(
                "Will use previously calculated X_test stored at", table_loc_test_saved
            )
            if debug:
                validate_feature_table(table_loc_test_saved, 8, "Test")
            return X_test, y_test
        else:
            print(
                table_loc_test_saved,
                "already exists, but no longer matches train data. Will re-make.",
            )
    print("Loading all feature tables for test...")
    test_files = tuple(glob(table_loc_test + "/*gzip"))
    X_test, y_test = sp.get_training_columns(table_filename=test_files)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print("Saving X_test to", table_loc_test_saved)
    X_test.to_parquet(table_loc_test_saved)
    if debug:
        validate_feature_table(table_loc_test_saved, 8, "Test")
    return X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cache",
        choices=["extract"] + [str(i) for i in range(4)],
        default="extract",
        help="Use option 'extract' to extract pre-built cache from tarball. You can also build the cache at runtime by specifying a cache building checkpoint(0-3), typically 0 or 3: 0 skips building the cache, 3 builds the entire cache.",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument(
        "-f",
        "--features",
        type=int,
        choices=range(17),
        default=16,
        help="Specify feature table building checkpoint(0-16), typically 0 or 16: 0 skips building feature tables, 16 builds all feature tables.",
    )
    parser.add_argument(
        "-fs",
        "--feature-selection",
        choices=["none", "skip", "yes"],
        default="yes",
        help="Option 'none' will use all features in subsequent steps, while 'skip' assumes this step has already been performed and will attempt to use its result in the following steps.",
    )
    parser.add_argument(
        "-n",
        "--n-jobs",
        type=int,
        default=1,
        help="Parameter will be used for sklearn modules with parallelization as appropriate.",
    )
    parser.add_argument(
        "-r",
        "--random-state",
        type=int,
        default=123,
        help="Random seed to use when needed in workflow.",
    )
    parser.add_argument(
        "-o",
        "--optimize",
        choices=["none", "skip", "yes"],
        default="yes",
        help="Option 'none' will not optimize hyperparameters and use mostly defaults, while 'skip' assumes this step has already been performed and will attempt to use its result in the following steps.",
    )
    parser.add_argument(
        "-tr",
        "--train-file",
        choices=["receptor_training.csv", "Mollentze_Training.csv"],
        default="Mollentze_Training.csv",
        help="File to be used corresponding to training data.",
    )
    parser.add_argument(
        "-ts",
        "--test-file",
        choices=["none", "Mollentze_Holdout.csv"],
        default="Mollentze_Holdout.csv",
        help="File to be used corresponding to test data.",
    )
    parser.add_argument(
        "-tc",
        "--target-column",
        choices=["Is_Both", "Is_Integrin", "Is_Sialic_Acid", "Human Host"],
        default="Human Host",
        help="Target column to be used for binary clasification.",
    )
    parser.add_argument(
        "-w",
        "--workflow",
        choices=["DTRA", "DR"],
        default="DR",
        help="Choice of machine learning workflow to be used.",
    )

    args = parser.parse_args()
    cache_checkpoint = args.cache
    debug = args.debug
    feature_checkpoint = args.features
    feature_selection = args.feature_selection
    n_jobs = args.n_jobs
    random_state = args.random_state
    optimize = args.optimize
    train_file = args.train_file
    test_file = args.test_file
    target_column = args.target_column
    workflow = args.workflow

    data = files("viral_seq.data")
    train_file = str(data.joinpath(train_file))
    test_file = str(data.joinpath(test_file))
    cache_file = str(data.joinpath("cache_mollentze.tar.gz"))
    viral_files = (
        [train_file, test_file]
        if test_file != str(data.joinpath("none"))
        else [train_file]
    )
    table_file = str(files("viral_seq.tests") / "train_test_table_info.csv")

    paths = []
    cache_extract_path = Path("data_external")
    extract_cookie = cache_extract_path / "CACHE_EXTRACTED_FROM_TARBALL"
    paths.append(cache_extract_path)
    paths.append(Path("data_external/cache_viral"))
    cache_viral = str(paths[-1])
    paths.append(Path("data_external/cache_isg"))
    cache_isg = str(paths[-1])
    paths.append(Path("data_external/cache_housekeeping"))
    cache_hk = str(paths[-1])

    paths.append(Path("data_calculated/tables/train"))
    table_loc_train = str(paths[-1])
    paths.append(Path("data_calculated/tables/test"))
    table_loc_test = str(paths[-1])
    table_locs = [table_loc_train, table_loc_test]
    paths.append(Path("data_calculated/tables/train_best"))
    table_loc_train_best = str(paths[-1] / "X_train.parquet.gzip")
    paths.append(Path("data_calculated/tables/test_saved"))
    table_loc_test_saved = str(paths[-1] / "X_test.parquet.gzip")
    hyperparams_path = Path("data_calculated/hyperparameters")
    paths.append(hyperparams_path)
    paths.append(Path("data_calculated/predictions"))
    predictions_loc = str(paths[-1] / "model_predictions.csv")
    model_path = Path("data_calculated/trained_models")
    paths.append(model_path)

    plots_path = Path("plots")
    paths.append(plots_path)
    optimization_plot_source = str(plots_path / "optimization_plot.csv")
    optimization_plot_figure = str(plots_path / "optimization_plot.png")
    feature_imp_consensus_plot_source = str(plots_path / "feat_imp_consensus.csv")
    feature_imp_consensus_plot_figure = str(plots_path / "feat_imp_consensus.png")

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    if debug:
        table_info = pd.read_csv(
            table_file,
            sep="\t",
            dtype={"Train_sum": np.float64, "Test_sum": np.float64},
            converters={
                "Train_shape": ast.literal_eval,
                "Test_shape": ast.literal_eval,
            },
        )

    build_cache(cache_checkpoint=cache_checkpoint, debug=debug)
    build_tables(feature_checkpoint=feature_checkpoint, debug=debug)
    X_train, y_train = feature_selection_rfc(
        feature_selection=feature_selection,
        debug=debug,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    X_test, y_test = get_test_features(
        table_loc_test,
        table_loc_test_saved,
        test_file,
        X_train,
        extract_cookie,
        debug=debug,
    )
    best_params: Dict[str, Any] = {}
    plotting_data: Dict[str, Any] = {}
    model_arguments: Dict[str, Any] = {}
    model_arguments = classifier.get_model_arguments(
        n_jobs,
        random_state,
        num_samples=X_train.shape[0],
        num_features=X_train.shape[1],
    )
    # optimize first if requested
    for name, val in model_arguments.items():
        params = val["optimize"]
        if optimize == "none":
            best_params[name] = {}
        else:
            print("===", name, "===")
            t_start = perf_counter()
            res = optimize_model(
                model=val["model"],
                X_train=X_train,
                y_train=y_train,
                optimize=optimize,
                debug=debug,
                random_state=random_state,
                name=name,
                n_jobs=n_jobs,
                outfile=str(hyperparams_path / ("params_" + val["suffix"] + ".json")),
                **params,
            )
            print(
                "Hyperparameter optimization of",
                name,
                "took",
                perf_counter() - t_start,
                "s",
            )
            best_params[name] = res["params"]
            plotting_data[name] = res["targets"]
    if optimize == "yes" or optimize == "skip":
        optimization_plots(
            plotting_data, optimization_plot_source, optimization_plot_figure
        )
    # train and predict on all models
    predictions = {}
    predictions["Species"] = pd.read_csv(test_file)["Species"]
    models_fitted = {}
    for name, val in model_arguments.items():
        models_fitted[name], predictions[name] = classifier.train_and_predict(
            val["model"],
            X_train,
            y_train,
            X_test,
            name=name,
            model_out=str(model_path / ("model_" + val["suffix"] + ".p")),
            params_predict=val["predict"],
            params_optimized=best_params[name],
        )
        this_auc = roc_auc_score(y_test, predictions[name])
        print(f"{name} achieved ROC AUC = {this_auc:.2f} on test data.")
    pd.DataFrame(predictions).to_csv(predictions_loc)
    # check feature importance and consensus
    feature_importances = []
    np.random.seed(random_state)  # used by `shap.summary_plot`
    for name, clf in models_fitted.items():
        print(f"Plotting feature importances for {name}")
        # built-in importances
        (
            sorted_feature_importances,
            sorted_feature_names,
        ) = feature_importance.sort_features(clf.feature_importances_, X_train.columns)
        feature_importance.plot_feat_import(
            sorted_feature_importances,
            sorted_feature_names,
            top_feat_count=10,
            model_name=name,
            fig_name_stem=str(
                plots_path / ("feat_imp_" + model_arguments[name]["suffix"])
            ),
        )
        feature_importances.append(clf.feature_importances_)
        # SHAP importances
        print("Calculating & Plotting SHAP values...")
        time_start = perf_counter()
        explainer = shap.Explainer(clf, seed=random_state)
        shap_values = explainer(X_train)
        print("Finished SHAP calculation in", perf_counter() - time_start)
        positive_shap_values = feature_importance.get_positive_shap_values(shap_values)
        feature_importance.plot_shap_meanabs(
            positive_shap_values,
            model_name=name,
            fig_name_stem=str(
                plots_path / f"feat_shap_mean_abs_{model_arguments[name]['suffix']}"
            ),
            top_feat_count=10,
        )
        feature_importance.plot_shap_beeswarm(
            positive_shap_values,
            model_name=name,
            fig_name=str(
                plots_path / f"feat_shap_beeswarm_{model_arguments[name]['suffix']}.png"
            ),
            max_display=10,
        )
        feature_importances.append(positive_shap_values)
    (
        ranked_feature_names,
        ranked_feature_counts,
        num_input_models,
    ) = feature_importance.feature_importance_consensus(
        pos_class_feat_imps=feature_importances,
        feature_names=X_train.columns,
        top_feat_count=10,
    )
    feature_importance.plot_feat_import_consensus(
        ranked_feature_names=ranked_feature_names,
        ranked_feature_counts=ranked_feature_counts,
        num_input_models=num_input_models,
        top_feat_count=10,
        fig_name=feature_imp_consensus_plot_figure,
        fig_source=feature_imp_consensus_plot_source,
    )

    X = pl.read_parquet(table_loc_train_best).to_pandas()
    tbl = pd.read_csv(train_file)
    y = tbl[target_column]

    random_state = np.random.RandomState(0)

    n_folds = 5
    cv = StratifiedKFold(n_splits=n_folds)
    clfr = RandomForestClassifier(
        n_estimators=10000, n_jobs=-1, random_state=random_state
    )
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(8, 8))

    counter = -1
    n_features = 5
    tmp1 = np.zeros((n_folds, n_features))

    for fold, (train, test) in enumerate(cv.split(X, y)):
        clfr.fit(X.iloc[train], y[train])
        df = pd.DataFrame()
        df["Features"] = X.iloc[train].columns
        df["Importances"] = clfr.feature_importances_
        df.sort_values(by=["Importances"], ascending=False, inplace=True)
        df.reset_index(inplace=True)
        print(df.iloc[:n_features])
        viz = RocCurveDisplay.from_estimator(
            clfr,
            X.iloc[test],
            y[test],
            name=f"ROC fold {fold + 1}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_folds - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        counter += 1
        tmp1[counter, :] = df["index"][:n_features].to_numpy()

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability\n(Positive label '"
        + str(target_column)
        + "')",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(paths[-1]) + "/" + "ROC_" + str(target_column) + ".png", dpi=300)

    (uniq, freq) = np.unique(tmp1.flatten(), return_counts=True)
    tmp2 = np.column_stack((uniq, freq))
    tmp3 = tmp2[tmp2[:, 1].argsort()]
    tmp4 = [
        df[df["index"] == tmp3[i, 0]]["Features"].to_numpy()
        for i in range(tmp3.shape[0])
    ]
    arr1 = tmp3[:, 1]
    arr2 = [tmp4[i][0] for i in range(len(tmp4))]
    fig, ax = plt.subplots(figsize=(8, 8))
    y_pos = np.arange(len(arr2))
    ax.barh(y_pos, (arr1 / n_folds) * 100)
    ax.set_xlim(0, 100)
    ax.set_yticks(y_pos, labels=arr2)
    ax.set_title(f"Feature importance consensus amongst {n_folds} folds")
    ax.set_xlabel("Percentage (%)")
    fig.tight_layout()
    fig.savefig(str(paths[-1]) + "/" + "FIC_" + str(target_column) + ".png", dpi=300)

    records = sp.load_from_cache(cache=cache_viral, filter=False)

    for ii in range(3):
        kmer_type = arr2[ii][5:7]
        kmer_specific = arr2[ii][8:]
        viruses = []
        products = []

        for record in records:
            for feature in record.features:
                if feature.type == "CDS":
                    nuc_seq = feature.location.extract(record.seq)
                    if len(nuc_seq) % 3 != 0:
                        continue
                    this_seq = nuc_seq.translate()
                    if kmer_type == "PC":
                        new_seq = ""
                        for each in this_seq:
                            if each in "AGV":
                                new_seq += "A"
                            elif each in "C":
                                new_seq += "B"
                            elif each in "FLIP":
                                new_seq += "C"
                            elif each in "MSTY":
                                new_seq += "D"
                            elif each in "HNQW":
                                new_seq += "E"
                            elif each in "DE":
                                new_seq += "F"
                            elif each in "KR":
                                new_seq += "G"
                            else:
                                new_seq += "*"
                        this_seq = new_seq

                    tmp5 = this_seq.find(kmer_specific)

                    if tmp5 > -1:
                        tmp6 = tbl.Accessions.isin([record.id])
                        if sum(tmp6):
                            viruses.append(tbl.Species[np.nonzero(tmp6)[0][0]])
                            products.append(feature.qualifiers["product"][0])

        print("Viruses: \n", viruses)
        print("Associated viral proteins: \n", products)

    if workflow == "DTRA":
    
        X = pl.read_parquet(table_loc_train_best).to_pandas()
        tbl = pd.read_csv(train_file)
        tmp = tbl
        tmp = tmp.drop(columns=['Citation_for_receptor','Whole_Genome','Mammal_Host','Primate_Host'])
        tmp = tmp.rename(columns={'Virus_Name': 'Species', 'Human_Host': 'Human Host', 'Receptor_Type': 'Is_Integrin'})
        tmp['Is_Sialic_Acid'] = pd.Series(dtype='bool')
        tmp['Is_Both'] = pd.Series(dtype='bool')
        tmp = tmp[['Species','Accessions','Human Host','Is_Integrin','Is_Sialic_Acid','Is_Both']]
        tmp['Is_Sialic_Acid'] = np.where(tmp['Is_Integrin'] == 'integrin', False, True)
        tmp['Is_Both'] = np.where(tmp['Is_Integrin'] == 'both', True, False)
        tmp['Is_Integrin'] = np.where(tmp['Is_Integrin'] == 'sialic_acid', False, True)
        tmp.to_csv(train_file)
        tbl = pd.read_csv(train_file)
        y = tbl[target_column]
        v1 = list(X.columns)
    
        random_state = np.random.RandomState(0)
    
        n_folds = 5
        cv = StratifiedKFold(n_splits=n_folds)
        clfr = RandomForestClassifier(
            n_estimators=10000, n_jobs=-1, random_state=random_state
        )
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
    
        fig, ax = plt.subplots(figsize=(8, 8))
    
        counter = -1
        n_features = 5
        tmp1 = np.zeros((n_folds, n_features))
    
        for fold, (train, test) in enumerate(cv.split(X, y)):
            clfr.fit(X.iloc[train], y[train])
            df = pd.DataFrame()
            df["Features"] = X.iloc[train].columns
            df["Importances"] = clfr.feature_importances_
            df.sort_values(by=["Importances"], ascending=False, inplace=True)
            df.reset_index(inplace=True)
            viz = RocCurveDisplay.from_estimator(
                clfr,
                X.iloc[test],
                y[test],
                name=f"ROC fold {fold + 1}",
                alpha=0.3,
                lw=1,
                ax=ax,
                plot_chance_level=(fold == n_folds - 1),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
    
            counter += 1
            tmp1[counter, :] = df["index"][:n_features].to_numpy()
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
    
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
    
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Mean ROC curve with variability\n(Positive label '"
            + str(target_column)
            + "')",
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(str(paths[-1]) + "/" + "ROC_" + str(target_column) + ".png", dpi=300)
    
        (uniq, freq) = np.unique(tmp1.flatten(), return_counts=True)
        tmp2 = np.column_stack((uniq, freq))
        tmp3 = tmp2[tmp2[:, 1].argsort()]
        tmp4 = [
            df[df["index"] == tmp3[i, 0]]["Features"].to_numpy()
            for i in range(tmp3.shape[0])
        ]
        arr1 = tmp3[:, 1]
        arr2 = [tmp4[i][0] for i in range(len(tmp4))]
        fig, ax = plt.subplots(figsize=(8, 8))
        y_pos = np.arange(len(arr2))
        ax.barh(y_pos, (arr1 / n_folds) * 100)
        ax.set_xlim(0, 100)
        ax.set_yticks(y_pos, labels=arr2)
        ax.set_title(f"Feature importance consensus amongst {n_folds} folds")
        ax.set_xlabel("Percentage (%)")
        fig.tight_layout()
        fig.savefig(str(paths[-1]) + "/" + "FIC_" + str(target_column) + ".png", dpi=300)
    
        explainer = shap.Explainer(clfr, seed=random_state)
        shap_values = explainer(X)
        positive_shap_values = shap_values[:, np.array(tmp3[:, 0],dtype=int), 1]
        fig, ax = plt.subplots(figsize=(8, 8))
        shap.summary_plot(positive_shap_values, show=False, plot_type="violin")
        fig, ax = plt.gcf(), plt.gca()
        ax.tick_params(labelsize=9)
        ax.set_title(f"Effect of Top {len(arr2)} Features for \n{i1}")
        fig.tight_layout()
        fig.savefig(fldr + str(i1.lower()) + '/plots/' + 'SHAP_' + str(i1) + '.png', dpi=300)
        
        records = sp.load_from_cache(cache=cache_viral, filter=False)
        
        viruses_AA = []
        proteins_AA = []
        start_loc_AA = []
        explicit_seq_AA = []
        kmers_AA = []
        viruses_PC = []
        proteins_PC = []
        start_loc_PC = []
        explicit_seq_AA_PC = []
        explicit_seq_PC_PC = []
        kmers_PC = []        
    
        for item in arr2:
            kmer_type = item[5:7]
            kmer_specific = item[8:]
            
            if kmer_type == 'AA':
                for record in records:
                    for feature in record.features:
                        if feature.type == 'CDS':
                            nuc_seq = feature.location.extract(record.seq)
                            if len(nuc_seq) % 3 != 0:
                                continue
                            this_seq_AA = nuc_seq.translate()
                            this_seq_AA = str(this_seq_AA)
    
                            v2 = [m.start() for m in re.finditer(kmer_specific,this_seq_AA)]
                    
                            if v2:
                                v3 = tbl.Accessions.isin([record.id])
                                if sum(v3):
                                    viruses_AA.append(tbl.Species[np.nonzero(v3)[0][0]])
                                    proteins_AA.append(feature.qualifiers['product'][0])
                                    start_loc_AA.append([v2])
                                    explicit_seq_AA.append(this_seq_AA) 
                                    kmers_AA.append(kmer_specific) 
    
            elif kmer_type == 'PC':
                for record in records:
                    for feature in record.features:
                        if feature.type == 'CDS':
                            nuc_seq = feature.location.extract(record.seq)
                            if len(nuc_seq) % 3 != 0:
                                continue
                            this_seq_AA = nuc_seq.translate()
                            new_seq = ''
                            for each in this_seq_AA:
                                if each in 'AGV':
                                    new_seq += 'A'
                                elif each in 'C':
                                    new_seq += 'B'
                                elif each in 'FLIP':
                                    new_seq += 'C'
                                elif each in 'MSTY':
                                    new_seq += 'D'
                                elif each in 'HNQW':
                                    new_seq += 'E'
                                elif each in 'DE':
                                    new_seq += 'F'
                                elif each in 'KR':
                                    new_seq += 'G'
                                else:
                                    new_seq += '*'
                            this_seq_PC = new_seq
                            this_seq_AA = str(this_seq_AA)
                            this_seq_PC = str(this_seq_PC)                              
    
                            v4 = [m.start() for m in re.finditer(kmer_specific,this_seq_PC)]
                            
                            if v4:
                                v5 = tbl.Accessions.isin([record.id])
                                if sum(v5):
                                    viruses_PC.append(tbl.Species[np.nonzero(v5)[0][0]])
                                    proteins_PC.append(feature.qualifiers['product'][0])
                                    start_loc_PC.append([v4])
                                    explicit_seq_AA_PC.append(this_seq_AA)                 
                                    explicit_seq_PC_PC.append(this_seq_PC)       
                                    kmers_PC.append(kmer_specific)                                           
    
        with open(str(data.joinpath('start_loc_PC.txt')), 'w') as file:
            for item in start_loc_PC:
                    file.write(" ".join(map(str,item)))
                    file.write("\n")
                    
        df = pd.DataFrame({'c1': kmers_PC,'c2': viruses_PC,'c3': proteins_PC})
        df.to_csv(str(data.joinpath('structural_unannotated.csv')), header=False, index=False)
