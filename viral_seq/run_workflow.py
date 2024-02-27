from importlib.resources import files
from viral_seq.analysis import spillover_predict as sp
from viral_seq.cli import cli
import pandas as pd
import re
import argparse
from http.client import IncompleteRead
from urllib.error import URLError
import polars as pl
import ast
from pytest import approx
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


def validate_feature_table(file_name, idx, prefix):
    print("Validating", file_name)
    df = pl.read_parquet(file_name).to_pandas()
    assert df.shape == table_info.iloc[idx][prefix + "_shape"]
    # check if numeric sum of the entire table matches what was precalculated
    assert df.select_dtypes(["number"]).to_numpy().sum() == approx(
        table_info.iloc[idx][prefix + "_sum"]
    )


def build_cache(cache_checkpoint=3, debug=False):
    """Download and store all data needed for the workflow"""

    if cache_checkpoint > 0:
        print("Will pull down data to local cache")

    if debug:
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
        # assert we don't have anything extra
        assert len(accessions_train.union(accessions_test)) == len(cache_subdirectories)

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
                    assert len(results) == 1
                    new_name = list(results)[0].split(".")[0]
                    print(accession, "was renamed as", new_name)
                    renamed_accessions[accession] = new_name
            for k, v in renamed_accessions.items():
                missing_hk.remove(k)
                extra_accessions.remove(v)
            assert len(extra_accessions) == 0
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
        # assert we don't have anything extra
        assert len(cached_accessions) + len(missing_hk) == len(accessions_hk)

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
        # there should only be one per directory, and they should be unique
        assert len(cache_subdir_set) == len(cache_subdirectories)
        missing = ensembl_ids.difference(cache_subdir_set)
        # assert there is nothing extra
        assert len(missing) + len(cache_subdir_set) == len(ensembl_ids)
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
        print("Debug mode: will run assertions on generated tables")

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
                    ],
                    standalone_mode=False,
                )
            if debug:
                idx = np.abs(8 - (this_checkpoint - this_checkpoint_modifier))
                validate_feature_table(this_outfile, idx, prefix)


def feature_selection_rfc(feature_selection, debug, n_jobs, random_state):
    """Sub-select features using best performing from a trained random forest classifier"""
    this_outfile_X = table_loc_train_best + "/X_train.parquet.gzip"
    this_outfile_y = table_loc_train_best + "/y_train.parquet.gzip"
    if feature_selection == "yes" or feature_selection == "none":
        print("Loading all feature tables for train...")
        train_files = tuple(glob(table_loc_train + "/*gzip"))
        X, y = sp.get_training_columns(table_filename=train_files)
        if feature_selection == "none":
            print(
                "All training features will be used as X_train in the following steps."
            )
        elif feature_selection == "yes":
            print(
                "Will train a random forest classifier to select the best performing features to use as X_train."
            )
            rfc = RandomForestClassifier(
                n_estimators=10_000, random_state=random_state, n_jobs=n_jobs
            )
            rfc.fit(X, y)
            sorted_imps, sorted_feats = zip(
                *sorted(zip(rfc.feature_importances_, rfc.feature_names_in_))
            )
            keep_feats = list(sorted_feats[-10_000:])
            X = X[keep_feats]
            print("Saving X_train, y_train to", this_outfile_X, "and", this_outfile_y)
            X.to_parquet(this_outfile_X)
            y.to_frame().to_parquet(this_outfile_y)
    elif feature_selection == "skip":
        print(
            "Will use previously calculated X_train, y_train stored at",
            this_outfile_X,
            "and",
            this_outfile_y,
        )
        X = pl.read_parquet(this_outfile_X).to_pandas()
        y = pl.read_parquet(this_outfile_y).to_pandas()["Human Host"]
    if debug:
        # these might not exist if the workflow has only been run with --feature-selection none
        if Path(this_outfile_X).is_file():
            validate_feature_table(this_outfile_X, 8, "Train")
        if Path(this_outfile_y).is_file():
            print("Validating", this_outfile_y)
            df_actual = pl.read_parquet(this_outfile_y).to_pandas()
            df_expected = pd.read_csv(train_file)[["Human Host"]]
            pd.testing.assert_frame_equal(df_actual, df_expected)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cache",
        type=int,
        choices=range(4),
        default=3,
        help="Specify cache building checkpoint(0-3), typically 0 or 3: 0 skips building the cache, 3 builds the entire cache.",
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

    args = parser.parse_args()
    cache_checkpoint = args.cache
    debug = args.debug
    feature_checkpoint = args.features
    feature_selection = args.feature_selection
    n_jobs = args.n_jobs
    random_state = args.random_state

    data = files("viral_seq.data")
    cache_viral = str(data / "cache_viral")
    train_file = str(data.joinpath("Mollentze_Training.csv"))
    test_file = str(data.joinpath("Mollentze_Holdout.csv"))
    viral_files = [train_file, test_file]
    cache_isg = str(data / "cache_isg")
    cache_hk = str(data / "cache_housekeeping")
    table_loc_train = str(data / "tables" / "train")
    table_loc_test = str(data / "tables" / "test")
    table_locs = [table_loc_train, table_loc_test]
    table_loc_train_best = str(data / "tables" / "train_best")
    table_file = str(files("viral_seq.tests") / "train_test_table_info.csv")
    if debug:
        table_info = pd.read_csv(
            table_file,
            sep="\t",
            dtype={"Train_sum": np.float32, "Test_sum": np.float32},
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
