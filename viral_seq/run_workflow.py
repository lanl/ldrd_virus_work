import tempfile
from importlib.resources import files
from viral_seq.analysis import spillover_predict as sp
from viral_seq.analysis import rfc_utils, classifier, feature_importance, get_features
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
from sklearn.metrics import roc_auc_score, RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from warnings import warn
import json
from typing import Dict, Any, Sequence, List
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import perf_counter
import tarfile
import shap
from collections import defaultdict
from scipy.stats import pearsonr

matplotlib.use("Agg")


def feature_signs(
    is_exposed: list,
    not_exposed: list,
    shap_values: np.ndarray,
    shap_data: np.ndarray,
) -> tuple:
    """
    Determine the sign character of the surface exposure status and response effect of the kmer
    based on the pearson correlation coefficient of the shap feature importance values

    Parameters:
    -----------
    is_exposed: list
        list of kmers that are surface exposed
    not_exposed: list
        list of kmers that are not surface exposed
    shap_values: np.ndarray
        shap feature importance values
    shap_data: np.ndarray
        shap expected values from dataset

    Returns:
    --------
    surface_exposed_sign: list
        list of +/- symbols denoting surface exposure status of kmer features in topN kmers
    response_effect: list
        list of +/- symbols denoting response effect from shap importance pearson-r correlation
    """
    response_effect = []
    surface_exposed_sign = []
    # add sign of surface exposure based on comparison between lists of exposure status
    for i in range(len(is_exposed)):
        sign = "x"
        if is_exposed[i]:
            sign = "+"
        elif not_exposed[i]:
            sign = "-"
        surface_exposed_sign.append(sign)
        # add sign of response effect based on pearson correllation coefficient
        pearson_r = pearsonr(
            shap_values[:, i],
            shap_data[:, i],
        )[0]
        if pearson_r > 0:
            response_effect.append("+")
        elif pearson_r <= 0:
            response_effect.append("-")

    return surface_exposed_sign, response_effect


def label_surface_exposed(
    list_of_kmers: Sequence[tuple], kmers_topN: list[str]
) -> tuple:
    """
    Creates lists of kmers denoting whether or not they are
    surface exposed ambiguous of PC or AA kmer status

    Parameters:
    -----------
    list_of_kmers: list
        list containing kmers and corresponding status of surface exposure as
        determined by cross-referencing top-N kmers with list of known surface
        exposed proteins

    kmers_topN: list
        list containing topN kmers from classifier

    Returns:
    --------
    is_exposed, not_exposed, found_kmers: tuple
        ordered lists of protein kmers that are either positively labeled as 'surface_exposed',
        negatively labeled as 'surface_exposed' or all kmers without PC/AA prefix in kmers_topN

    """

    # make lists 'is_exposed', 'not_exposed', found_kmers by cross-
    # referencing surface exposed status with protein kmers list
    is_exposed = [x[0] for x in list_of_kmers if x[1] == "Yes"]
    not_exposed = [x[0] for x in list_of_kmers if x[1] == "No"]
    found_kmers = [
        x.replace("kmer_PC_", "").replace("kmer_AA_", "") for x in kmers_topN
    ]

    # and re-order lists to align with the list of found kmers
    is_exposed = [s if s in is_exposed else "" for s in found_kmers]
    not_exposed = [s if s in not_exposed else "" for s in found_kmers]

    return is_exposed, not_exposed, found_kmers


def FIC_plot(
    topN_kmers: list,
    kmer_count: np.ndarray,
    n_folds: int,
    target_column: str,
    exposure_status_sign: list,
    response_effect_sign: list,
    path: Path,
):
    """
    Create Feature Importance Consensus (FIC) plot from SHAP explainer values and
    native estimator feature importances using important kmers found during ML
    classification. Include the following information for each kmer as labels on the plot:
        1. the sign of the pearson correlation coefficient for the feature importance values (+/-)
        2. the status of the kmer as surface exposed or not surface exposed based on the known virus (+/-)

    Parameters:
    -----------
    topN_kmers: list
        list of top_N kmers from ML classifiers
    kmer_count: array
        consensus number of classifer folds agreeing upon corresponding kmer in topN_kmers
    n_folds: int
        number of cv folds performed by classifier
    target_column: str
        name of column on which classifier was trained
    exposure_status_sign: list
        list of +/- symbols denoting surface exposure status of kmer features in topN kmers
    response_effect_sign: list
        list of +/- symbols denoting response effect from shap importance pearson-r correlation
    path: Path
        path to save figure
    """

    if target_column == "Is_Integrin":
        target_name = "Integrin"
    elif target_column == "Is_Sialic_Acid":
        target_name = "Sialic Acid"
    elif target_column == "Is_Both":
        target_name = "Integrin and Sialic Acid"
    else:
        target_name = target_column

    fig, ax = plt.subplots(figsize=(8, 8))
    y_pos = np.arange(len(topN_kmers))
    ax.barh(y_pos, (kmer_count / n_folds) * 100, color="k")
    ax.set_xlim(0, 100)
    ax.set_yticks(y_pos, labels=topN_kmers)
    ax.set_title(
        f"Feature importance consensus amongst {n_folds} folds\n for {target_name} binding"
    )
    ax.set_xlabel("Classifier Consensus Percentage (%)")
    kmer_idx = -1

    for p in ax.patches:
        kmer_idx += 1
        left, bottom, width, height = p.get_bbox().bounds

        if response_effect_sign[kmer_idx] == "+":
            response_sign_color = "g"
        elif response_effect_sign[kmer_idx] == "-":
            response_sign_color = "r"

        if exposure_status_sign[kmer_idx] == "+":
            exposure_sign_color = "g"
        elif exposure_status_sign[kmer_idx] == "-":
            exposure_sign_color = "r"
        else:
            exposure_sign_color = "y"

        ax.annotate(
            response_effect_sign[kmer_idx],
            xy=(left + width / 4, bottom + height / 2),
            ha="center",
            va="center",
            color=response_sign_color,
            fontsize="xx-large",
        )
        ax.annotate(
            exposure_status_sign[kmer_idx],
            xy=(left + 3 * width / 4, bottom + height / 2),
            ha="center",
            va="center",
            color=exposure_sign_color,
            fontsize="xx-large",
        )

        plus_symbol = Line2D(
            [0], [0], marker="+", color="green", markersize=9, linestyle=""
        )
        minus_symbol = Line2D(
            [0], [0], marker="_", color="red", markersize=9, linestyle=""
        )
        cross_symbol = Line2D(
            [0], [0], marker="x", color="gold", markersize=9, linestyle=""
        )

        ax.legend(
            handles=[
                plus_symbol,
                minus_symbol,
                plus_symbol,
                minus_symbol,
                cross_symbol,
            ],
            labels=[
                "on left: Positive effect on response",
                "on left: Negative effect on response",
                "on right: Protein is surface-exposed",
                "on right: Protein is not surface-exposed",
                "on right: Protein does not bind to both\n receptor types",
            ],
            loc="lower right",
            prop={"size": 9},
        )

        fig.tight_layout()
        fig.savefig(str(path) + "/" + "FIC_" + str(target_column) + ".png", dpi=300)
        plt.close()


def csv_conversion(input_csv: str = "receptor_training.csv") -> pd.DataFrame:
    """
    Convert the original version of the input CSV table
    (i.e., that is merged to main) to a new DataFrame stored in memory
    (i.e., that is compatible with the ML workflow)

    Parameters
    ----------
    input_csv : str
        A string representing the name of the input table that will undergo conversion.
        Default is `receptor_training.csv`.

    Returns
    -------
    output_df : pd.DataFrame
        A Pandas DataFrame representing the converted form of the input table
        that accounts for the target column(s) of interest.

    """

    table = pd.read_csv(input_csv)

    table = table.drop(
        columns=["Citation_for_receptor", "Whole_Genome", "Mammal_Host", "Primate_Host"]
    )
    table = table.rename(
        columns={
            "Virus_Name": "Species",
            "Human_Host": "Human Host",
            "Receptor_Type": "Is_Integrin",
        }
    )
    table["Is_Both"] = np.where(table["Is_Integrin"] == "both", True, False)
    table["Is_Sialic_Acid"] = pd.Series(dtype="bool")
    table = table[
        [
            "Species",
            "Accessions",
            "Human Host",
            "Is_Integrin",
            "Is_Sialic_Acid",
            "Is_Both",
        ]
    ]
    table["Is_Sialic_Acid"] = np.where(table["Is_Integrin"] == "integrin", False, True)
    table["Is_Integrin"] = np.where(table["Is_Integrin"] == "sialic_acid", False, True)
    return table


def check_positive_controls(
    positive_controls: Sequence[str],
    kmers_list: list[str],
    mapping_method: str,
    mode: str,
) -> pd.DataFrame:
    """
    checks how many of the kmers that are known to bind to a specific surface
    receptor (positive controls) are found in a given list of kmers (train vs. top-N)

    Parameters
    ----------
    positive_controls: list
        A list of strings denoting known AA-kmers associated with surface protein binding
    kmers_list: list
        A list containing kmer feature strings from either the training dataset
        or topN classifier rankings
    mapping_method: str
        preference for mapping AA-kmers to PC-kmers
    mode: str
        type of kmer to check for in dataset (AA vs. PC) to avoid double counting

    Returns:
    --------
    kmers_df: pd.DataFrame
        Dataframe containing the names of kmer features found in the dataset that contain
        each of the positive control sequences, and the counts for each
    """

    if mode == "PC":
        # map positive_controls to PC-kmers using desired mapping method
        pc_pos_con = []
        for p_con in positive_controls:
            pc_kmer = ""
            for each in p_con:
                pc_kmer += get_features.aa_map(each, method=mapping_method)
            pc_pos_con.append(pc_kmer)
        positive_controls = pc_pos_con

    # iterate through lists of positive controls and count number
    # of positive controls found in kmer_list/add kmers to dictionary
    kmer_counts: dict[str, int] = {key: 0 for key in positive_controls}
    kmers_out: dict[str, List[str]] = {key: [] for key in positive_controls}
    for kmer_feat in kmers_list:
        if kmer_feat.startswith(f"kmer_{mode}_"):
            kmer_str = kmer_feat[8:]
            for positive_control in positive_controls:
                if positive_control in kmer_str:
                    kmer_counts[positive_control] += 1
                    kmers_out[positive_control].append(kmer_feat)

    kmers_out_df = pd.DataFrame.from_dict(kmers_out, orient="index").transpose()
    kmer_counts_df = pd.DataFrame.from_dict(kmer_counts, orient="index").transpose()
    kmers_df = pd.concat(
        [kmers_out_df, kmer_counts_df], ignore_index=True
    ).convert_dtypes()

    return kmers_df


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
        if feature_checkpoint > 0:
            for i, (file, folder) in enumerate(zip(viral_files, table_locs)):
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                    csv_conversion(file).to_csv(temp_file)
                    file = temp_file.name
                if i == 0:
                    prefix = "Train"
                    debug = False
                    this_outfile = folder + "/" + prefix + "_main.parquet.gzip"
                    if feature_checkpoint > 5:
                        feature_checkpoint = 5
                    this_checkpoint = feature_checkpoint

                for k in range(11 - feature_checkpoint, 11):
                    this_outfile = folder + "/" + prefix + "_k{}.parquet.gzip".format(k)
                    if feature_checkpoint >= this_checkpoint:
                        print(
                            "Building table for Train",
                            "which includes kmers and pc kmers with k={}.".format(k),
                        )
                        print(
                            "To restart at this point use --features",
                            this_checkpoint,
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
                                "--mapping-method",
                                mapping_method,
                            ],
                            standalone_mode=False,
                        )
                        this_checkpoint -= 1


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
        choices=["Is_Integrin", "Is_Sialic_Acid", "Is_Both", "Human Host"],
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
    parser.add_argument(
        "-m",
        "--mapping-method",
        choices=["shen_2007", "jurgen_schmidt"],
        default="shen_2007",
        help="Preference of scheme for mapping AA-kmers to PC-kmers",
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
    train_file = args.train_file
    test_file = args.test_file
    target_column = args.target_column
    workflow = args.workflow
    mapping_method = args.mapping_method

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
    if workflow == "DR":
        X_test, y_test = get_test_features(
            table_loc_test,
            table_loc_test_saved,
            test_file,
            X_train,
            extract_cookie,
            debug=debug,
        )
        best_params: Dict[str, Any] = {}
        best_params_group: Dict[str, Any] = {}
        plotting_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        model_arguments: Dict[str, Any] = {}
        rng = np.random.default_rng(random_state)
        # sklearn only accepts random_state in the range [0, 4294967295]; uint32
        random_states = rng.integers(
            np.iinfo(np.uint32).max, dtype=np.uint32, size=copies
        )
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
                this_filename = "params_" + val["group"] + ".json"
                with open(
                    str(hyperparams_stored_path.joinpath(this_filename)), "r"
                ) as f:
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
                    outfile=str(
                        hyperparams_path / ("params_" + val["suffix"] + ".json")
                    ),
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
                name=name,
                model_out=str(model_path / ("model_" + val["suffix"] + ".p")),
                params_predict=val["predict"],
                params_optimized=best_params[name],
            )

            this_auc = roc_auc_score(y_test, predictions[val["group"]][name])
            print(f"{name} achieved ROC AUC = {this_auc:.2f} on test data.")
        # ROC curve plotting
        comp_names: list[str] = []
        comp_fprs: list[Any] = []
        comp_tprs: list[Any] = []
        comp_tpr_stds: list[Any] = []
        for group in predictions:
            fpr, tpr, tpr_std = classifier.get_roc_curve(y_test, predictions[group])
            this_title = (
                f"ROC Curve\nAveraged over {copies} seeds"
                if copies > 1
                else "ROC Curve"
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
            # output all predictions to .csv
            predictions[group]["Species"] = pd.read_csv(test_file)["Species"]
            pd.DataFrame(predictions[group]).to_csv(
                str(predictions_path / (group + "_predictions.csv"))
            )
        this_title = (
            f"ROC Curve\nEach model averaged over {copies} seeds"
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
        # check feature importance and consensus
        feature_importances = []
        np.random.seed(random_state)  # used by `shap.summary_plot`
        for name, clf in models_fitted.items():
            print(f"Plotting feature importances for {name}")
            (
                sorted_feature_importances,
                sorted_feature_names,
            ) = feature_importance.sort_features(
                clf.feature_importances_, X_train.columns
            )
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
            positive_shap_values = feature_importance.get_positive_shap_values(
                shap_values
            )
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
                    plots_path
                    / f"feat_shap_beeswarm_{model_arguments[name]['suffix']}.png"
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

    elif workflow == "DTRA":
        records = sp.load_from_cache(cache=cache_viral, filter=False)

        # Here is a list of common integrin-binding motifs. These motifs interact with
        # specific integrins, playing critical roles in cell adhesion, signaling, and
        # interaction with the extracellular matrix:
        # RGD (Arg-Gly-Asp)
        # KGE (Lys-Gly-Glu)
        # LDV (Leu-Asp-Val)
        # DGEA (Asp-Gly-Glu-Ala)
        # REDV (Arg-Glu-Asp-Val)
        # YGRK (Tyr-Gly-Arg-Lys)
        # PHSRN (Pro-His-Ser-Arg-Asn)
        # SVVYGLR (Ser-Val-Val-Tyr-Gly-Leu-Arg)

        pos_controls = [
            "RGD",
            "KGE",
            "LDV",
            "DGEA",
            "REDV",
            "YGRK",
            "PHSRN",
            "SVVYGLR",
        ]

        ### Create empty lists for eventual post-processing of data output

        viruses_PC = []
        protein_name_PC = []
        k_mers_PC = []

        X = pl.read_parquet(table_loc_train_best).to_pandas()
        tbl = csv_conversion(train_file)
        y = tbl[target_column]

        ### Estimation and visualization of the variance of the
        ### Receiver Operating Characteristic (ROC) metric using cross-validation.
        ### Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

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
        temp1 = np.zeros((n_folds, n_features))

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
            temp1[counter, :] = df["index"][:n_features].to_numpy()

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
        fig.savefig(
            str(paths[-1]) + "/" + "ROC_" + str(target_column) + ".png", dpi=300
        )
        plt.close(fig)

        ### Populate 'array1' and 'array2' with useful information
        ### for the Feature Importance Consensus (FIC) and SHAP plots

        (uniq, freq) = np.unique(temp1.flatten(), return_counts=True)
        temp2 = np.column_stack((uniq, freq))
        temp3 = temp2[temp2[:, 1].argsort()]
        temp4 = [
            df[df["index"] == temp3[i, 0]]["Features"].to_numpy()
            for i in range(temp3.shape[0])
        ]
        array1 = temp3[:, 1]
        array2 = [temp4[i][0] for i in range(len(temp4))]

        # count of PC positive controls in train data
        pos_con_train_PC = check_positive_controls(
            positive_controls=pos_controls,
            kmers_list=list(X.iloc[train].columns),
            mapping_method=mapping_method,
            mode="PC",
        )
        print(
            "Count of Positive Control PC k-mers in Train Dataset:\n",
            pos_con_train_PC.tail(1).to_string(index=False),
        )
        pos_con_train_PC.to_csv(
            f"train_data_PC_kmer_positive_controls_{mapping_method}.csv",
            na_rep="",
            index=False,
        )

        # count of PC positive controls in topN (array2)
        pos_con_topN_PC = check_positive_controls(
            positive_controls=pos_controls,
            kmers_list=array2,
            mapping_method=mapping_method,
            mode="PC",
        )
        print(
            "Count of Positive Control PC k-mers in topN:\n",
            pos_con_topN_PC.tail(1).to_string(index=False),
        )
        pos_con_topN_PC.to_csv(
            f"topN_PC_kmer_positive_controls_{mapping_method}.csv",
            na_rep="",
            index=False,
        )

        # count of AA positive controls in train data
        pos_con_train_AA = check_positive_controls(
            positive_controls=pos_controls,
            kmers_list=list(X.iloc[train].columns),
            mapping_method=mapping_method,
            mode="AA",
        )
        print(
            "Count of Positive Control AA k-mers in Train Dataset:\n",
            pos_con_train_AA.tail(1).to_string(index=False),
        )
        pos_con_train_AA.to_csv(
            f"train_data_AA_kmer_positive_controls_{mapping_method}.csv",
            na_rep="",
            index=False,
        )

        # count of AA positive controls in topN (array2)
        pos_con_topN_AA = check_positive_controls(
            positive_controls=pos_controls,
            kmers_list=array2,
            mapping_method=mapping_method,
            mode="AA",
        )
        print(
            "Count of Positive Control AA k-mers in TopN:\n",
            pos_con_topN_AA.tail(1).to_string(index=False),
        )
        pos_con_train_AA.to_csv(
            f"topN_AA_kmer_positive_controls_{mapping_method}.csv",
            na_rep="",
            index=False,
        )

        for (
            item
        ) in (
            array2
        ):  # 'array2' contains only PC-kmers; all of the AA-kmers just so happen to be filtered out by the Random Forest Classifier
            k_mer = item[8:]

            for record in records:
                for feature in record.features:
                    if feature.type == "CDS":
                        nuc_seq = feature.location.extract(record.seq)
                        if len(nuc_seq) % 3 != 0:
                            continue
                        this_seq_AA = nuc_seq.translate()
                        this_seq_AA = str(this_seq_AA)

                    if (
                        feature.type == "mat_peptide"
                    ):  # using help from Tyler's code in the comments to MR !49 on GitLab
                        new_seq = ""

                        for each in this_seq_AA:
                            new_seq += get_features.aa_map(each, method=mapping_method)
                        this_seq_PC = new_seq
                        this_seq_PC = str(this_seq_PC)

                        v2 = [m.start() for m in re.finditer(k_mer, this_seq_PC)]

                        if v2:
                            v3 = tbl.Accessions.isin([record.id])
                            if sum(v3):
                                viruses_PC.append(tbl.Species[np.nonzero(v3)[0][0]])
                                protein_name_PC.append(
                                    str(feature.qualifiers.get("product"))[2:-2]
                                )
                                k_mers_PC.append(k_mer)

        # manually curated on the basis of output from `np.unique(protein_name_PC)`
        surface_exposed = [
            "1B(VP2)",
            "1C(VP3)",
            "1D(VP1)",
            "Envelope surface glycoprotein gp120",
            "PreM protein",
            "VP1",
            "VP1 protein",
            "VP2",
            "VP2 protein",
            "VP3",
            "VP3 protein",
            "envelope glycoprotein E1",
            "envelope glycoprotein E2",
            "envelope protein",
            "envelope protein E",
            "membrane glycoprotein M",
            "membrane glycoprotein precursor prM",
            "membrane protein M",
        ]
        # list comprehension
        surface_exposed_status = [
            "Yes" if item in surface_exposed else "No" for item in protein_name_PC
        ]
        # manually curated on the basis of links (DOIs and ViralZone urls) that I could find for a subset of items in `np.unique(protein_name_PC)`. The curation of references is currently incomplete.
        references = [
            "membrane protein M",
            "1B(VP2)",
            "1C(VP3)",
            "1D(VP1)",
            "Envelope surface glycoprotein gp120",
            "3C",
            "3C protein",
            "3D",
            "3D protein",
            "3D-POL protein",
            "Hel protein",
            "Lab protein",
            "Lb protein",
            "1A(VP4)",
            "nucleocapsid",
            "p1",
            "p2",
            "p6",
            "p66 subunit",
            "p7 protein",
            "pre-membrane protein prM",
            "protein VP0",
            "protein pr",
            "protien 3A",
            "protein 1A",
            "protein 1B",
            "protein 1C",
            "protein 1D",
            "protein 2A",
            "protein 2B",
            "protein 2C",
            "protien 2K",
            "protein 3A",
            "protein 3AB",
            "protein 3C",
            "protein 3D",
        ]
        urls = [
            "https://doi.org/10.1099/0022-1317-69-5-1105",
            "https://doi.org/10.3389/fmicb.2020.562768",
            "https://doi.org/10.3389/fmicb.2020.562768",
            "https://doi.org/10.3389/fmicb.2020.562768",
            "https://doi.org/10.1038/31405",
            "https://doi.org/10.3390/v15122413",
            "https://doi.org/10.3390/v15122413",
            "https://doi.org/10.3389/fimmu.2024.1365521",
            "https://doi.org/10.3389/fimmu.2024.1365521",
            "https://doi.org/10.3389/fimmu.2024.1365521",
            "https://doi.org/10.1016/j.virusres.2024.199401",
            "https://doi.org/10.1128/jvi.74.24.11708-11716.2000",
            "https://doi.org/10.1128/jvi.74.24.11708-11716.2000",
            "https://doi.org/10.3389/fmicb.2020.562768",
            "https://doi.org/10.1007/s11904-011-0107-3",
            "https://doi.org/10.1007/s11904-011-0107-3",
            "https://doi.org/10.1007/s11904-011-0107-3",
            "https://doi.org/10.1007/s11904-011-0107-3",
            "https://doi.org/10.1002/cbic.202000263",
            "https://doi.org/10.1038/s41598-019-44413-x",
            "https://doi.org/10.1016/0042-6822(92)90267-S",
            "https://doi.org/10.1128/jvi.73.11.9072-9079.1999",
            "https://doi.org/10.1042/BJ20061136",
            "https://doi.org/10.1128/jvi.00791-17",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
            "https://viralzone.expasy.org/99",
        ]
        refs_urls_dict = dict(zip(references, urls))
        # list comprehension
        citations = [
            refs_urls_dict[item] if item in references else "missing"
            for item in protein_name_PC
        ]

        temp5 = list(set(zip(k_mers_PC, surface_exposed_status)))
        res1, res2, res3 = label_surface_exposed(temp5, array2)

        ### Production of the annotated CSV file with information for each case of `target_column`
        df = pd.DataFrame(
            {
                "Virus (corresponding to PC k-mer)": viruses_PC,
                "Protein (corresponding to virus)": protein_name_PC,
                "Status of protein (surface-exposed or not)": surface_exposed_status,
                "Citation corresponding to protein status": citations,
            }
        )
        df.to_csv("annotated_" + str(target_column) + ".csv", header=True, index=False)

        ### Production of the SHAP plot

        explainer = shap.Explainer(clfr, seed=random_state)
        shap_values = explainer(X)
        positive_shap_values = shap_values[:, np.array(temp3[:, 0][::-1], dtype=int), 1]
        feature_importance.plot_shap_beeswarm(
            positive_shap_values,
            model_name="Random Forest",
            fig_name=str(paths[-1]) + "/" + "SHAP_" + str(target_column) + ".png",
            max_display=len(array2),
        )

        # build lists of feature exposure and response effect signs for FIC plotting
        exposure_status_sign, response_effect_sign = feature_signs(
            res1, res2, positive_shap_values.values, positive_shap_values.data
        )

        # Production of the FIC plot
        FIC_plot(
            array2,
            array1,
            n_folds,
            target_column,
            exposure_status_sign,
            response_effect_sign,
            paths[-1],
        )
