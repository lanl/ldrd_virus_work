import json
import pickle
from pathlib import Path
from viral_seq.analysis.get_features import get_genomic_features, get_kmers, get_gc
from tqdm import tqdm
from Bio import Entrez, SeqIO
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
)
from urllib.request import HTTPError
import time

matplotlib.use("Agg")


def _append_recs(record_folder):
    # genbank format has more metadata we can use so
    # focus on that for now; eventually we might assert
    # that the FASTA file has the same sequence read in
    # as sanity check
    genbank_file = list(record_folder.glob("*.genbank"))[0]
    with open(genbank_file) as genfile:
        record = list(SeqIO.parse(genfile, "genbank"))[0]
    return record


def run_search(
    search_terms: list[str],
    retmax: int,
    email: str,
    must: bool = False,
    attempts: int = 3,
) -> set[str]:
    Entrez.email = email
    acc_set: set[str] = set()
    for search_term in search_terms:
        # try to prevent timeouts on large search queries
        # by batching
        search_batch_size = min(10_000, retmax)
        count = 0
        remaining = retmax
        print(f"starting Entrez esearch (search batch size is: {search_batch_size})")
        print("Search term:", search_term)
        for retstart in tqdm(range(0, retmax, search_batch_size)):
            if remaining < search_batch_size:
                actual_retmax = remaining
            else:
                actual_retmax = search_batch_size
            # We should handle network exceptions
            for attempt in range(attempts):
                try:
                    handle = Entrez.esearch(
                        db="nucleotide",
                        term=search_term,
                        retstart=retstart,
                        retmax=actual_retmax,
                        idtype="acc",
                        usehistory="y",
                    )
                except HTTPError as err:
                    # Wait and try again
                    print(
                        "Attempt", attempt + 1, "/", attempts, "failed with error:", err
                    )
                    # We might want to enforce that this succeeds
                    if attempt == attempts - 1 and must:
                        raise
                    time.sleep(1)
                else:
                    search_results = Entrez.read(handle)
                    acc_set = acc_set.union(search_results["IdList"])
                    count += int(search_results["Count"])
                    remaining -= search_batch_size
                    handle.close()
                    break
        print("after Entrez esearch")

    return acc_set


def load_results(acc_set: set[str], email: str, must: bool = False, attempts: int = 3):
    Entrez.email = email
    records = []
    batch_size = min(100, len(acc_set))
    numrecs = len(acc_set)
    print(f"fetching {numrecs} Entrez records with batch_size {batch_size}:")
    acc_list = list(acc_set)
    for start in tqdm(range(0, numrecs, batch_size)):
        # make a few attempts and handle exceptions
        for attempt in range(attempts):
            try:
                handle = Entrez.efetch(
                    db="nuccore",
                    rettype="gb",
                    id=acc_list[start : start + batch_size],
                    idtype="acc",
                    retmode="text",
                )
            except HTTPError as err:
                # Wait and try again
                print("Attempt", attempt + 1, "/", attempts, "failed with error:", err)
                # require success if must
                if attempt == attempts - 1 and must:
                    raise
                time.sleep(1)
            else:
                records += list(SeqIO.parse(handle, "gb"))
                handle.close()
                break
    return records


def init_cache(folder=".cache"):
    cache_path = Path(folder)
    # envisioned local sequence cache layout:
    # .cache/ folder
    # subfolders are formatted as accession numbers
    # and each subfolder should contain the FASTA and GENBANK
    # sequences for that accession number

    if not cache_path.exists():
        cache_path.mkdir(parents=False, exist_ok=False)
    cache_subdirectories = [x for x in cache_path.iterdir() if x.is_dir()]
    return cache_path, cache_subdirectories


def filter_records(
    records, just_assert: bool = False, just_warn: bool = False, verbose: bool = True
):
    # Checks the records satisfy the conditions defined
    # If just_assert, it will throw an exception if a record is bad
    # If just_warn, it will print warnings for bad records but take no action
    # Otherwise, it will return the records that pass

    # TODO: current work flow uses hand-picked sequences so these filters are very lenient. Will need stronger filters if workflow changes

    filtered_records = []
    for record in tqdm(records, disable=not verbose):
        # we also sanity check each record before
        # trying to cache it locally;
        # each record is a Bio.SeqRecord.SeqRecord
        len(record)
        msg = ""
        if record.description.startswith("Homo sapiens"):
            msg += "Record is human in origin. "
        if "partial" in record.description:
            msg += "Record is not a complete genome. "
        if record.description.endswith("mRNA"):
            msg += "Record description ends with mRNA. "
        if "Klebsiella" in record.annotations["organism"]:
            # this should help prevent the sequence that got pulled in
            # for: https://re-git.lanl.gov/treddy/ldrd_virus_work/-/issues/13
            # TODO: if we expand to include i.e., DNA viral genomes, we'll
            # almost certainly need more sophisticated filtering than just
            # one bacterial species though
            msg += "Organism is Klebsiella. "
        if len(msg) > 0:
            if just_warn:
                print("Warning:", record.id, msg)
            elif just_assert:
                raise AssertionError(record.id, msg)
            else:
                continue
        if not just_assert and not just_warn:
            filtered_records.append(record)
    if not just_assert and not just_warn:
        print(len(filtered_records), "retained of", len(records), "checked")
        return filtered_records


def add_to_cache(records, cache: str = ".cache"):
    cache_path, cache_subdirectories = init_cache(cache)
    # with the records list populated from the online
    # search, we next want to grow the local cache with
    # folders + FASTA/GENBANK format files that are not
    # already there
    print(
        "performing quality analysis of sequences and adding to local cache if they pass..."
    )
    records = filter_records(records)
    num_recs_added = 0
    num_recs_excluded = 0
    for record in tqdm(records):
        record_cache_path = cache_path / f"{record.id}"
        if record_cache_path not in cache_subdirectories:
            record_cache_path.mkdir(parents=False, exist_ok=False)
            with open(record_cache_path / f"{record.id}.fasta", "w") as fasta_file:
                SeqIO.write(record, fasta_file, "fasta")
            with open(record_cache_path / f"{record.id}.genbank", "w") as genbank_file:
                SeqIO.write(record, genbank_file, "genbank")
            num_recs_added += 1
        else:
            num_recs_excluded += 1
    print(
        f"number of records added to the local cache from online search: {num_recs_added}"
    )
    print(
        f"number of records excluded from the online search because they were already present: {num_recs_excluded}"
    )


def load_from_cache(accessions=None, cache: str = ".cache", verbose: bool = True):
    # TODO: Verbose flag added as this call prints too much in certain circumstances but should implement a proper verbose mode
    cache_path, cache_subdirectories = init_cache(cache)
    directories_to_load = []
    # TODO: if there is nothing in the cache, fail
    if accessions is None:
        if verbose:
            print("Loading entire cache")
        directories_to_load = cache_subdirectories[:]
    else:
        if verbose:
            print("Will attempt to load selected accessions from local cache")
        for accession in accessions:
            this_record_cache_path = cache_path / f"{accession}"
            # we can only load what exists
            if this_record_cache_path in cache_subdirectories:
                directories_to_load.append(this_record_cache_path)
    if verbose:
        print(
            "total number of records (sequence folders) to load from local cache:",
            len(directories_to_load),
        )
        print("loading the local records cache (genetic sequences) into memory")
    records = []
    for record_folder in tqdm(directories_to_load, disable=not verbose):
        records.append(_append_recs(record_folder))
    # TODO: if we change data retention filters when scraping, we will possibly need to update the cache
    if verbose:
        print("Asserting records loaded from cache conform to quality standards...")
    filter_records(records, just_assert=True, verbose=verbose)
    return records


def _grab_features(features, records, genomic, kmers, kmer_k, gc):
    feat_genomic = None
    feat_kmers = None
    feat_gc = None
    if genomic:
        feat_genomic = get_genomic_features(records)
        if feat_genomic is None:
            return None
    if kmers:
        feat_kmers = get_kmers(records, k=kmer_k)
        if feat_kmers is None:
            return None
    if gc:
        feat_gc = get_gc(records)
        if feat_gc is None:
            return None
    if feat_genomic is not None:
        features.update(feat_genomic)
    if feat_kmers is not None:
        features.update(feat_kmers)
    if feat_gc is not None:
        features.update(feat_gc)
    return features


def build_table(
    df,
    rfc=None,
    save: bool = False,
    filename: str = "df.parquet.gzip",
    cache: str = ".cache",
    genomic: bool = True,
    kmers: bool = True,
    kmer_k: int = 10,
    gc: bool = True,
    ordered: bool = True,
):
    # make a list of all the accessions and a dict to keep track of which species an accession belongs to
    records_dict: dict[str, list] = {}
    accessions_dict: dict[str, str] = {}
    row_dict: dict[str, pd.Series] = {}
    accessions = []
    for index, row in df.iterrows():
        records_dict[row["Species"]] = []
        row_dict[row["Species"]] = row
        for accession in row["Accessions"].split():
            accessions.append(accession)
            accessions_dict[accession] = row["Species"]
    calculated_feature_rows = []
    # we do one call to load all records from cache
    records_unordered = load_from_cache(accessions, cache=cache, verbose=False)
    for record in records_unordered:
        records_dict[accessions_dict[record.id]].append(record)
    meta_data = list(df.columns)
    for species, records in tqdm(records_dict.items()):
        features = row_dict[species].to_dict()
        this_result = _grab_features(features, records, genomic, kmers, kmer_k, gc)
        if this_result is not None:
            calculated_feature_rows.append(this_result)
    table = pd.DataFrame.from_records(calculated_feature_rows)
    if rfc is not None:
        # add columns from training if missing
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [[0] * len(rfc.feature_names)],
                    index=[-1],
                    columns=rfc.feature_names,
                ),
            ]
        )
        table.drop(index=-1, inplace=True)
        # only retain columns from training
        table = table[meta_data + rfc.feature_names]
    else:
        # drop a column if 1 or less row has a value
        # this should only affect kmers
        table.dropna(axis=1, thresh=2, inplace=True)
    table.fillna(0, inplace=True)
    # required for repeatability, but may be slow
    if ordered:
        table.sort_values(by=["Unnamed: 0"], inplace=True)
        table = table.reindex(sorted(table.columns), axis=1)
    table.reset_index(drop=True, inplace=True)
    if save:
        print(
            "Saving the pandas DataFrame of genomic data to a parquet file:", filename
        )
        table.to_parquet(filename, engine="pyarrow", compression="gzip")
    return table


def get_training_columns(
    df=None, class_column: str = "Human Host", table_filename: str = ""
):
    # Check if we are using data from file or a passed DataFrame
    if table_filename == "" and df is None:
        raise ValueError("No data provided to train random forest model.")
    elif df is None:
        print("Loading the pandas DataFrame from a parquet file:", table_filename)
        df = pd.read_parquet(table_filename, engine="fastparquet")
    elif table_filename == "":
        print("Using provided DataFrame")
    else:
        raise ValueError("Ambiguous request; both DataFrame and file passed.")
    if class_column not in df.columns:
        raise ValueError("Can't find class column to train random forest model.")
    # Any columns not dropped here are assumed to be features
    df_features = df.drop(class_column, axis=1)
    df_features.drop(
        columns=["index", "Species", "Accessions", "Unnamed: 0"],
        errors="ignore",
        inplace=True,
    )
    if len(df_features.columns) < 1:
        raise ValueError("Data contains no features.")
    elif len(df_features.columns) == 1:
        X = df_features.to_numpy().reshape(-1, 1)
    else:
        X = df_features
    y = df[class_column]
    return X, y


def train_rfc(
    X,
    y,
    save: bool = False,
    filename: str = "random_forest_model.p",
    **kwargs,
):
    # kwargs override default hyperparameters
    # TODO: add other tunable hyperparameters
    hype = {
        "estimators": 70,
        "rand": 123,
    }
    for key in hype:
        if key in kwargs:
            hype[key] = kwargs[key]
    rfc = RandomForestClassifier(
        n_estimators=hype["estimators"], random_state=hype["rand"]
    )
    rfc.fit(X, y)
    # keep track of what features we trained
    if isinstance(X, pd.DataFrame):
        rfc.feature_names = list(X.columns)
    if save:
        print("Saving random forest model to file:", filename)
        with open(filename, "wb") as outfile:
            pickle.dump(rfc, outfile)
    return rfc


def predict_rfc(
    X,
    y,
    rfc=None,
    filename: str = "random_forest_model.p",
    plot: bool = False,
    out_prefix: str = "",
    file_predict_proba: str = "predictions.csv",
    file_roc_curve: str = "roc_curve.csv",
    file_metrics: str = "metrics.json",
):
    file_predict_proba = out_prefix + file_predict_proba
    file_roc_curve = out_prefix + file_roc_curve
    file_metrics = out_prefix + file_metrics
    if rfc is None:
        with open(filename, "rb") as rfc_file:
            rfc = pickle.load(rfc_file)
    # Add features from Random Forest Classifier that may be missing from validation features
    # This should be redundant
    if isinstance(X, pd.DataFrame):
        X = pd.concat(
            [
                X,
                pd.DataFrame(
                    [[0] * len(rfc.feature_names)],
                    index=[-1],
                    columns=rfc.feature_names,
                ),
            ]
        )
        X.drop(index=-1, inplace=True)
        X.fillna(0, inplace=True)
        # Only predict on features trained in Random Forest Classifier
        y_scores = rfc.predict_proba(X[rfc.feature_names])
    else:
        y_scores = rfc.predict_proba(X)
    y_scores = y_scores[:, 1].reshape(-1, 1)
    this_auc = roc_auc_score(y.astype(bool), y_scores)
    if plot:
        # individual predictions
        # TODO: in order to check which prediction corresponds to which virus, you need to cross-reference the data table (if they are in the same order)
        df_predictions = pd.DataFrame()
        df_predictions["ground_truth"] = y
        df_predictions["probability"] = y_scores
        df_predictions.to_csv(file_predict_proba)
        # ROC curve for plotting
        df_roc_curve = pd.DataFrame()
        fpr, tpr, thresholds = roc_curve(y, y_scores)
        df_roc_curve["fpr"] = fpr
        df_roc_curve["tpr"] = tpr
        df_roc_curve["thresholds"] = thresholds
        df_roc_curve.to_csv(file_roc_curve)
        # This file should be used to store any metrics of interest
        with open(file_metrics, "w") as f:
            json.dump({"AUC": this_auc}, f)

    return this_auc


def cross_validation(
    X,
    y,
    splits=5,
    plot: bool = False,
    prefix="cv_",
    **kwargs,
):
    # TODO: this function should be expanded to allow for hyperparameter search
    cv = StratifiedKFold(n_splits=splits)
    aucs = []
    for fold, (train, test) in enumerate(cv.split(X, y)):
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train]
            X_test = X.iloc[test]
        else:
            X_train = X[train]
            X_test = X[test]
        rfc = train_rfc(X_train, y[train], **kwargs)
        this_prefix = prefix + str(fold) + "_"
        this_auc = predict_rfc(
            X_test, y[test], rfc=rfc, plot=plot, out_prefix=this_prefix
        )
        aucs.append(this_auc)
    return aucs


def plot_roc(roc_files, filename="roc_plot.png", title="ROC curve"):
    """Basic ROC plotting function with little customization for rapid visualization needs"""
    # TODO: if there are many plots, such functions should be relocated
    fig, ax = plt.subplots(figsize=(6, 6))
    mean_fpr = np.linspace(0, 1, 100)
    aucs = []
    tprs = []
    # Fold plots
    for i, file in enumerate(roc_files):
        df = pd.read_csv(file)
        this_auc = auc(df["fpr"], df["tpr"])
        aucs.append(this_auc)
        label = r"Fold %d (AUC = %0.2f)" % (i, this_auc)
        ax.plot(df["fpr"], df["tpr"], label=label)
        interp_tpr = np.interp(mean_fpr, df["fpr"], df["tpr"])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    if len(roc_files) > 1:
        # mean plot
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
        # +/- std plot
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

    # chance
    ax.plot([0, 1], [0, 1], "r--")

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.axis("square")
    ax.legend(loc="lower right")

    fig.savefig(filename)
