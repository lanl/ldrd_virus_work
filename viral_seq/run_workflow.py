from importlib.resources import files
from viral_seq.analysis import spillover_predict as sp
from viral_seq.analysis import rfc_utils, classifier, feature_importance
from viral_seq.cli import cli
import viral_seq.data.make_data_summary_plots as data_summary
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
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
from warnings import warn
import json
from typing import Dict, Any
import matplotlib
import matplotlib.pyplot as plt
from time import perf_counter
import tarfile
import shap
from collections import defaultdict
import os
from ray.tune.search.sample import Domain

matplotlib.use("Agg")


def validate_feature_table(file_name, idx, prefix, loose=False):
    # rtol loosened to account for lib version discrepancies
    # so far this is only seen to affect tables after feature selection
    atol = 6 if loose else 0
    rtol = 1e-3 if loose else 1e-8
    print("Validating", file_name)
    df = pl.read_parquet(file_name).to_pandas()
    expected_shape = table_info.iloc[idx][prefix + "_shape"]
    if not df.shape[0] == expected_shape[0]:
        raise ValueError(
            "Number of rows in feature table does not match what was precalculated.\nActual: %s\nExpected: %s"
            % (df.shape[0], expected_shape[0])
        )
    if not np.allclose(df.shape[1], expected_shape[1], rtol=0, atol=atol):
        raise ValueError(
            "Number of columns in feature table is not within acceptable variation from what was precalculated.\nActual: %s\nExpected: %s"
            % (df.shape[1], expected_shape[1])
        )
    actual_sum = df.select_dtypes(["number"]).to_numpy().sum()
    expected_sum = table_info.iloc[idx][prefix + "_sum"]
    if not np.allclose(actual_sum, expected_sum, rtol=rtol):
        raise ValueError(
            "The feature table's numerical sum, which sums all feature values for all viruses, is not within acceptable variation from what was precalculated. Verify integrity of input data and feature calculation.\nActual: %s\nExpected: %s"
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


def build_tables(feature_checkpoint=0, debug=False, target_column="Human Host"):
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

    # tables are built in multiple parts as this is faster for reading/writing
    tables_per_dataset = 19
    for i, (file, folder) in enumerate(zip(viral_files, table_locs)):
        prefix = "Train" if i == 0 else "Test"
        this_checkpoint_modifier = (1 - i) * tables_per_dataset
        this_checkpoint = tables_per_dataset + this_checkpoint_modifier
        this_outfile = folder + "/" + prefix + "_main.parquet.gzip"
        if feature_checkpoint >= this_checkpoint:
            print(
                "Building table for",
                prefix,
                "which includes genomic features, gc content, and similarity features for ISG and housekeeping genes.",
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
                    "--similarity-genomic",
                    "--similarity-cache",
                    cache_isg + " " + cache_hk,
                    "--target-column",
                    target_column,
                ],
                standalone_mode=False,
            )
        if debug:
            idx = np.abs(
                tables_per_dataset - (this_checkpoint - this_checkpoint_modifier)
            )
            validate_feature_table(this_outfile, idx, prefix)
        for kmer_type, kmer_suff in [("AA", ""), ("PC", "-pc")]:
            for k in range(2, 11):
                this_checkpoint -= 1
                this_outfile = folder + "/" + prefix + f"_k{kmer_type}{k}.parquet.gzip"
                if feature_checkpoint >= this_checkpoint:
                    print(
                        f"Building table for {prefix} which includes {kmer_type} kmers with k={k}.",
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
                            f"--features-kmers{kmer_suff}",
                            f"--kmer-k{kmer_suff}",
                            str(k),
                            "--target-column",
                            target_column,
                        ],
                        standalone_mode=False,
                    )
                if debug:
                    idx = np.abs(
                        tables_per_dataset
                        - (this_checkpoint - this_checkpoint_modifier)
                    )
                    validate_feature_table(this_outfile, idx, prefix)


def feature_selection_rfc(
    feature_selection, debug, n_jobs, random_state, target_column="Human Host"
):
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
            validate_feature_table(table_loc_train_best, 19, "Train", loose=True)
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
        # use untunable parameters used during optimization for baseline
        # other parameters will not be specified (use defaults)
        default_config = {k: v for k, v in config.items() if not isinstance(v, Domain)}
        default_config["n_jobs"] = n_jobs
        default_score = classifier.cv_score(
            model, X=X_train, y=y_train, **default_config
        )
        print("ROC AUC with default settings:", default_score)
        res["targets"] = [default_score] + res["targets"]
        if default_score > res["target"]:
            print(
                "Will use default settings since we weren't able to improve with optimization."
            )
            res["target"] = default_score
            res["params"] = default_config
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


def optimization_plots(input_data: Dict[str, Any], name_prefix: str, plots_path: Path):
    out_source = str(plots_path / (name_prefix + "_optimization_plot.csv"))
    out_fig = str(plots_path / (name_prefix + "_optimization_plot.png"))
    print("Writing optimization plot data to", out_source)
    for name, targets in input_data.items():
        target_max = np.maximum.accumulate(targets)
        df = pd.DataFrame(target_max, columns=[name])
        if Path(out_source).is_file():
            old_df = pd.read_csv(out_source)
            df = pd.concat([old_df.loc[:, list(old_df.columns != name)], df], axis=1)
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
    plt.close()


def get_test_features(
    table_loc_test,
    table_loc_test_saved,
    test_file,
    X_train,
    extract_cookie,
    debug=False,
    target_column="Human Host",
):
    if not extract_cookie.is_file():
        debug = False
    print("Ensuring X_test has only the features in X_train...")
    if Path(table_loc_test_saved).exists():
        X_test = pl.read_parquet(table_loc_test_saved).to_pandas()
        y_test = pd.read_csv(test_file)[target_column]
        if set(X_test.columns) == set(X_train.columns):
            print(
                "Will use previously calculated X_test stored at", table_loc_test_saved
            )
            if debug:
                validate_feature_table(table_loc_test_saved, 19, "Test", loose=True)
            return X_test, y_test
        else:
            print(
                table_loc_test_saved,
                "already exists, but no longer matches train data. Will re-make.",
            )
    print("Loading all feature tables for test...")
    test_files = tuple(glob(table_loc_test + "/*gzip"))
    X_test, y_test = sp.get_training_columns(
        table_filename=test_files, class_column=target_column
    )
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    print("Saving X_test to", table_loc_test_saved)
    X_test.to_parquet(table_loc_test_saved)
    if debug:
        validate_feature_table(table_loc_test_saved, 19, "Test", loose=True)
    return X_test, y_test


def _plot_confusion_matrices(
    y_test: np.ndarray,
    model_arguments: Dict[str, Any],
    predictions_ensemble_hard_eer: Dict[str, np.ndarray],
    comp_names_ensembles: list[str],
    comp_preds_ensembles: list[np.ndarray],
    plots_path: Path,
):
    predictions_group = defaultdict(list)
    # individual classifiers
    for name, val in model_arguments.items():
        group = val["group"]
        # for group confusion matrix
        predictions_group[group].append(predictions_ensemble_hard_eer[name])
        classifier.plot_confusion_matrix(
            y_test,
            predictions_ensemble_hard_eer[name],
            title=f"Confusion Matrix for Test Dataset\n{name}\npredictions at EER threshold",
            filename=str(plots_path / f"{name}_confusion_matrix.png"),
        )
    copies = len(predictions_group[group])
    # groups mean +/- std
    if copies > 1:
        for group in predictions_group:
            classifier.plot_confusion_matrix_mean(
                y_test,
                predictions_group[group],
                title=f"Average Confusion Matrix for Test Dataset\n{group}\npredictions at EER thresholds across {copies} seeds",
                filename=str(plots_path / f"{group}_confusion_matrix.png"),
            )
    # ensembles
    for this_name, this_ensemble in zip(
        comp_names_ensembles, comp_preds_ensembles, strict=True
    ):
        classifier.plot_confusion_matrix(
            y_test,
            this_ensemble,
            title=f"Confusion Matrix for Test Dataset\n{this_name}",
            filename=str(plots_path / f"{this_name}_confusion_matrix.png"),
        )


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
        choices=range(39),
        default=38,
        help="Specify feature table building checkpoint(0-38), typically 0 or 38: 0 skips building feature tables, 38 builds all feature tables.",
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
        choices=["none", "skip", "yes", "pre-optimized"],
        default="pre-optimized",
        help="Default option 'pre-optimized' uses parameters previously calculated by the authors, while 'yes' will run the same optimization procedure but the results are not deterministic due to parallelization. Option 'skip' assumes this step has been performed previously with option 'yes' and will attempt to use that result in the following steps. Option 'none' will not optimize hyperparameters and use mostly defaults.",
    )
    parser.add_argument(
        "-cp",
        "--copies",
        type=int,
        default=1,
        help="Select the number of copies of each model to use. Each copy will have a different seed chosen deterministically by '--random-state'",
    )
    parser.add_argument(
        "-co",
        "--check-optimization",
        action="store_true",
        help="Run hyperparameter optimization for every copy to compare how optimization is affected by seed.",
    )
    parser.add_argument(
        "-cc",
        "--check-calibration",
        action="store_true",
        help="Calibrate classifiers with `CalibratedClassifierCV`.",
    )
    parser.add_argument(
        "-tr",
        "--train-file",
        choices=[
            "Mollentze_Training.csv",
            "Mollentze_Training_Fixed.csv",
            "Mollentze_Training_Shuffled.csv",
            "Relabeled_Train.csv",
            "Relabeled_Train_Human_Shuffled.csv",
            "Relabeled_Train_Mammal_Shuffled.csv",
            "Relabeled_Train_Primate_Shuffled.csv",
        ],
        default="Mollentze_Training.csv",
        help="File to be used corresponding to training data.",
    )
    parser.add_argument(
        "-ts",
        "--test-file",
        choices=[
            "Mollentze_Holdout.csv",
            "Mollentze_Holdout_Fixed.csv",
            "Mollentze_Holdout_Shuffled.csv",
            "Relabeled_Test.csv",
            "Relabeled_Test_Human_Shuffled.csv",
            "Relabeled_Test_Mammal_Shuffled.csv",
            "Relabeled_Test_Primate_Shuffled.csv",
        ],
        default="Mollentze_Holdout.csv",
        help="File to be used corresponding to test data.",
    )
    parser.add_argument(
        "-tc",
        "--target-column",
        choices=["Human Host", "human", "mammal", "primate"],
        default="Human Host",
        help="Target column to be used for binary classification.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
    )

    args = parser.parse_args()
    cache_checkpoint = args.cache
    debug = args.debug
    feature_checkpoint = args.features
    feature_selection = args.feature_selection
    n_jobs = args.n_jobs
    random_state = args.random_state
    optimize = args.optimize
    copies = args.copies
    check_optimization = args.check_optimization
    check_calibration = args.check_calibration
    train_file = args.train_file
    test_file = args.test_file
    target_column = args.target_column
    no_summary = args.no_summary

    if debug and (
        train_file != "Mollentze_Training.csv" or test_file != "Mollentze_Holdout.csv"
    ):
        raise ValueError(
            "Debug Mode is intended to validate the workflow by running checks with the default training (Mollentze_Training.csv) and testing (Mollentze_Holdout.csv) data only."
        )
    data = files("viral_seq.data")
    train_file = str(data.joinpath(train_file))
    test_file = str(data.joinpath(test_file))
    cache_file = str(data.joinpath("cache_mollentze.tar.gz"))
    viral_files = [train_file, test_file]
    table_file = str(files("viral_seq.tests.expected") / "train_test_table_info.csv")
    hyperparams_stored_path = files("viral_seq.data.hyperparameters")

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
    predictions_path = Path("data_calculated/predictions")
    paths.append(predictions_path)
    model_path = Path("data_calculated/trained_models")
    paths.append(model_path)

    plots_path = Path("plots")
    paths.append(plots_path)
    feature_imp_consensus_plot_source = str(plots_path / "feat_imp_consensus.csv")
    feature_imp_consensus_plot_figure = str(plots_path / "feat_imp_consensus.png")
    family_heatmap_plot_source = str(plots_path / "family_heatmap.csv")
    family_heatmap_plot_figure = str(plots_path / "family_heatmap.png")

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

    if not no_summary:
        data_summary.plot_family_heatmap(
            train_file,
            test_file,
            target_column,
            family_heatmap_plot_figure,
            family_heatmap_plot_source,
        )
        relative_entropy = data_summary.relative_entropy_viral_families(
            heatmap_csv=family_heatmap_plot_source
        )
        print(
            f"Relative entropy of viral family distribution between train and test datasets: {relative_entropy:.3f}"
        )

    build_cache(cache_checkpoint=cache_checkpoint, debug=debug)
    build_tables(
        feature_checkpoint=feature_checkpoint, debug=debug, target_column=target_column
    )
    X_train, y_train = feature_selection_rfc(
        feature_selection=feature_selection,
        debug=debug,
        n_jobs=n_jobs,
        random_state=random_state,
        target_column=target_column,
    )
    X_test, y_test = get_test_features(
        table_loc_test,
        table_loc_test_saved,
        test_file,
        X_train,
        extract_cookie,
        debug=debug,
        target_column=target_column,
    )
    best_params: Dict[str, Any] = {}
    best_params_group: Dict[str, Any] = {}
    plotting_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
    model_arguments: Dict[str, Any] = {}
    rng = np.random.default_rng(random_state)
    # sklearn only accepts random_state in the range [0, 4294967295]; uint32
    random_states = rng.integers(np.iinfo(np.uint32).max, dtype=np.uint32, size=copies)
    for rs in random_states:
        model_arguments.update(
            classifier.get_model_arguments(
                n_jobs,
                rs,
                num_samples=X_train.shape[0],
                num_features=X_train.shape[1],
            )
        )
    # optimize first if requested
    for name, val in model_arguments.items():
        params = val["optimize"]
        if optimize == "none":
            best_params[name] = {}
        elif val["group"] in best_params_group:
            best_params[name] = best_params_group[val["group"]]
        elif optimize == "pre-optimized":
            print("Using hyperparameters pre-caclulated for", val["group"])
            train_file_suff = os.path.splitext(os.path.basename(train_file))[0]
            if train_file_suff == "Relabeled_Train":
                # in the non-shuffled and relabeled case, cached hyperparameter files follow a different naming convention
                if "human" not in target_column.lower():
                    train_file_suff = f"{train_file_suff}_{target_column}"
                else:
                    # the cached hyperparameter files for human
                    # target don't have a _human suffix
                    train_file_suff = f"{train_file_suff}"
            this_filename = f"params_{val['group']}_{train_file_suff}.json"
            with open(str(hyperparams_stored_path.joinpath(this_filename)), "r") as f:
                res = json.load(f)
            best_params[name] = res["params"]
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
            if not check_optimization:
                print(
                    f"All other copies of {val['group']} will use the same parameters."
                )
                best_params_group[val["group"]] = res["params"]
            plotting_data[val["group"]][name] = res["targets"]
    if optimize == "yes" or optimize == "skip":
        for group, this_data in plotting_data.items():
            optimization_plots(this_data, group, plots_path)
    # cross-validation
    test_folds: list = []
    predictions_cv = defaultdict(list)
    model_eer_threshold: dict[str, np.float64] = defaultdict(np.float64)
    for name, val in model_arguments.items():
        these_params = {
            k: v for k, v in best_params[name].items() if k not in val["predict"]
        }
        cv_data = classifier.cross_validation(
            val["model"],
            X_train,
            y_train,
            **val["predict"],
            **these_params,
        )
        if len(test_folds):
            assert np.allclose(
                np.hstack(cv_data.y_tests), np.hstack(test_folds)
            ), "`classifier.cross_validation` is not returning identical test sets for each model/copy"
        else:
            test_folds = cv_data.y_tests
        predictions_cv[val["group"]].append(cv_data.y_preds)
        # get EER data
        eer_data_cv = []
        fprs = []
        tprs = []
        thresholds = []
        for i, pred in enumerate(cv_data.y_preds):
            this_fpr, this_tpr, this_thresh = roc_curve(test_folds[i], pred)
            fprs.append(this_fpr)
            tprs.append(this_tpr)
            this_eer_data = classifier.cal_eer_thresh_and_val(
                this_fpr, this_tpr, this_thresh
            )
            eer_data_cv.append(this_eer_data)
            thresholds.append(this_eer_data.eer_threshold)
        model_eer_threshold[name] = np.mean(thresholds)
        classifier.plot_roc_curve_comparison(
            [f"Fold {i}" for i in range(len(test_folds))],
            fprs,
            tprs,
            filename=str(plots_path / f"{name.replace(' ', '_')}_eer_roc_plot.png"),
            title=f"ROC Curve\n{name}\nCross Validation on Training",
            eer_data_list=eer_data_cv,
        )
    assert int(len(model_arguments) / copies) == len(
        predictions_cv
    ), f"Number of cross-validation predictions doesn't match number of model types: {len(predictions_cv)} != {int(len(model_arguments)/copies)}"
    comp_names_cv: list[str] = []
    comp_fprs_cv: list[Any] = []
    comp_tprs_cv: list[Any] = []
    for group in predictions_cv:
        cv_roc_data = classifier.get_roc_curve_cv(test_folds, predictions_cv[group])
        this_title = (
            f"ROC Curve\nCross-validation on Training\nAveraged over {copies} seeds"
            if copies > 1
            else "ROC Curve\nCross-validation on Training"
        )
        these_names = [f"{group} fold {i}" for i in range(len(cv_roc_data.tpr_folds))]
        classifier.plot_roc_curve_comparison(
            these_names,
            cv_roc_data.fpr_folds,
            cv_roc_data.tpr_folds,
            cv_roc_data.tpr_std_folds,
            filename=str(plots_path / f"{group}_roc_plot_cv.png"),
            title=this_title,
        )
        comp_names_cv.append(group)
        comp_tprs_cv.append(cv_roc_data.mean_tpr)
    this_title = (
        f"ROC Curve\nCross-validation on Training\nAveraged over 5 folds and {copies} seeds"
        if copies > 1
        else "ROC Curve\nCross-validation on Training\nAveraged over 5 folds"
    )
    classifier.plot_roc_curve_comparison(
        comp_names_cv,
        None,
        comp_tprs_cv,
        filename=str(plots_path / "CV_roc_plot_comparison.png"),
        title=this_title,
    )
    # train and predict on all models
    predictions: Dict[str, Dict[str, Any]] = defaultdict(dict)
    models_fitted = {}
    for name, val in model_arguments.items():
        (
            models_fitted[name],
            predictions[val["group"]][name],
        ) = classifier.train_and_predict(
            val["model"],
            X_train,
            y_train,
            X_test,
            y_test,
            name=name,
            model_out=str(model_path / ("model_" + val["suffix"] + ".p")),
            params_predict=val["predict"],
            params_optimized=best_params[name],
            calibrate=check_calibration,
            filename_calibration_curve=str(
                plots_path / f"{name.replace(' ', '_')}_calibration_curve.png"
            ),
        )
    # Statistics and plotting with test predictions
    comp_names: list[str] = []
    comp_fprs: list[Any] = []
    comp_tprs: list[Any] = []
    comp_tpr_stds: list[Any] = []
    for group in predictions:
        fpr, tpr, tpr_std = classifier.get_roc_curve(y_test, predictions[group])
        this_title = (
            f"ROC Curve\nAveraged over {copies} seeds" if copies > 1 else "ROC Curve"
        )
        classifier.plot_roc_curve(
            group,
            fpr,
            tpr,
            tpr_std,
            filename=str(plots_path / f"{group}_roc_plot.png"),
            title=this_title,
        )
        comp_names.append(group)
        comp_fprs.append(fpr)
        comp_tprs.append(tpr)
        if copies > 1:
            y_proba_group = np.array(list(predictions[group].values())).mean(axis=0)
            classifier.plot_calibration_curve(
                y_test,
                y_proba_group,
                title=f"Calibration Curve\n{group}\nAveraged over {copies} seeds",
                filename=str(plots_path / f"{group}_calibration_curve.png"),
            )
        # output all predictions to .csv
        predictions[group]["Species"] = pd.read_csv(test_file)["Species"]
        pd.DataFrame(predictions[group]).to_csv(
            str(predictions_path / (group + "_predictions.csv"))
        )
        predictions[group].pop("Species")
    this_title = (
        f"ROC Curve\nEach classifier type averaged over {copies} seeds."
        if copies > 1
        else "ROC Curve"
    )
    classifier.plot_roc_curve_comparison(
        comp_names,
        comp_fprs,
        comp_tprs,
        filename=str(plots_path / "roc_plot_comparison.png"),
        title=this_title,
    )
    # ensemble models
    comp_names_ensembles = []
    comp_preds_ensembles = []
    comp_fprs_ensembles = []
    comp_tprs_ensembles = []
    predictions_ensemble_hard = {}
    predictions_ensemble_hard_eer = {}
    for group in predictions:
        for name in predictions[group]:
            # compare EER threshold to typical hard votes 0.5 threshold
            predictions_ensemble_hard[name] = predictions[group][name] > 0.5
            predictions_ensemble_hard_eer[name] = (
                predictions[group][name] > model_eer_threshold[name]
            )
    for name, these_predictions in {
        "0.5": predictions_ensemble_hard,
        "EER": predictions_ensemble_hard_eer,
    }.items():
        df_ortho = pd.DataFrame.from_dict(these_predictions)
        this_ensemble = df_ortho.mode(axis="columns").iloc[:, 0]  # type: ignore
        this_ensemble_proba = df_ortho.sum(axis=1) / df_ortho.shape[1]
        this_fpr, this_tpr, this_thresh = roc_curve(y_test, this_ensemble_proba)
        df_out = pd.DataFrame(this_ensemble_proba, columns=["Ensemble Averaged Votes"])
        df_out["Species"] = pd.read_csv(test_file)["Species"]
        df_out.to_csv(
            str(predictions_path / f"ensemble_hard_{name}_proba_predictions.csv")
        )
        comp_names_ensembles.append(f"Hard Votes Ensemble at {name} Threshold")
        comp_preds_ensembles.append(this_ensemble)
        comp_fprs_ensembles.append(this_fpr)
        comp_tprs_ensembles.append(this_tpr)
    this_title = (
        f"ROC Curve\nEnsemble Models of {len(predictions_ensemble_hard)} independent classifiers\n({len(predictions)} model types across {copies} seeds)"
        if copies > 1
        else "ROC Curve\nEnsemble Models"
    )
    models_for_ensembles = [(k, v) for k, v in models_fitted.items()]
    comp_names_thresh_ensembles = comp_names_ensembles.copy()
    pred_stack, fpr_stack, tpr_stack = classifier._ensemble_stacking_logistic(
        models_for_ensembles,
        X_train,
        y_train,
        X_test,
        y_test,
        test_file,
        predictions_path,
        plots_path,
        cv="prefit",
        plot_weights=copies == 1,
        estimator_names=list(predictions.keys()),
    )
    comp_names_ensembles.append("StackingClassifier")
    comp_names_thresh_ensembles.append("StackingClassifier at 0.5 Threshold")
    comp_preds_ensembles.append(pred_stack)
    comp_fprs_ensembles.append(fpr_stack)
    comp_tprs_ensembles.append(tpr_stack)
    classifier.plot_roc_curve_comparison(
        comp_names_ensembles,
        comp_fprs_ensembles,
        comp_tprs_ensembles,
        filename=str(plots_path / "roc_plot_ensemble_comparison.png"),
        title=this_title,
    )
    _plot_confusion_matrices(
        y_test,
        model_arguments,
        predictions_ensemble_hard_eer,
        comp_names_thresh_ensembles,
        comp_preds_ensembles,
        plots_path,
    )
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
        shap_values = feature_importance.get_shap_values(clf, X_train, random_state)
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
