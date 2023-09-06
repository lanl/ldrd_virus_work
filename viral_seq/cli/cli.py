import rich_click as click
import viral_seq.analysis.spillover_predict as sp
import pandas as pd
import pickle
from functools import partial


# Shared options
_option_email = partial(
    click.option,
    "--email",
    show_default=True,
    default="arhall@lanl.gov",
    help=(
        "Email used when communicating with Pubmed. "
        "You may be contacted if requests are deemed excessive. "
    ),
)

_option_cache = partial(
    click.option,
    "--cache",
    "-c",
    default=".cache",
    show_default=True,
    help=("Specify local cache to use for this command. "),
)

_option_file = partial(
    click.option,
    "--file",
    "-f",
    required=True,
    help=("Provide a file to use for this command. "),
)

_option_rfc_file = partial(
    click.option,
    "--rfc-file",
    default=None,
    required=True,
    help=("Pickle file of a RandomForestClassifier to be used for this command. "),
)
_option_outfile = partial(
    click.option,
    "--outfile",
    "-o",
    show_default=True,
    help=("Name of output file to be saved for this command. "),
)

_option_features_genomic = partial(
    click.option,
    "--features-genomic",
    "-g",
    is_flag=True,
    help=("Calculate viral genomic features, including: ..."),
)

_option_features_gc = partial(
    click.option,
    "--features-gc",
    "-gc",
    is_flag=True,
    help=("Calculate GC content feature. "),
)

_option_features_kmers = partial(
    click.option,
    "--features-kmers",
    "-kmers",
    is_flag=True,
    help=("Calculate amino acid Kmer features."),
)

_option_kmers_k = partial(
    click.option,
    "--kmer-k",
    "-k",
    default=10,
    show_default=True,
    help=("K value to use for amino acid Kmer calculation."),
)

_option_prefix = partial(
    click.option,
    "--prefix",
    "-p",
    default="cli_",
    show_default=True,
    help=("Prefix prepended to output files."),
)


@click.group()
def cli():
    pass


# --- search-data ---
@cli.command()
@_option_email()
@_option_cache()
@click.option(
    "--query", "-q", required=True, help=("Search used on Pubmed to pull records.")
)
@click.option(
    "--retmax",
    "-r",
    default=1,
    show_default=True,
    help=("Maximum number of records to store locally."),
)
def search_data(email, query, retmax, cache):
    """Scrape data results of a search term from Pubmed and store locally for further use."""
    search_terms = query.split()
    results = sp.run_search(search_terms=search_terms, retmax=retmax, email=email)
    records = sp.load_results(results, email=email)
    sp.add_to_cache(records, cache=cache)


# --- pull-data ---
@cli.command()
@_option_email()
@_option_cache()
@_option_file(help=("Provide a .csv file that contains an 'Accessions' column. "))
def pull_data(email, cache, file):
    """Retrieve accessions from Pubmed and store locally for further use."""
    df = pd.read_csv(file)
    if "Accessions" not in df:
        raise ValueError("Provided .csv must contain an Accessions column.")
    # Entries in 'Accessions' column may be space delimited accessions for species with multiple segments
    accessions = set((" ".join(df["Accessions"].values)).split())
    records = sp.load_results(accessions, email=email)
    sp.add_to_cache(records, cache=cache)


# --- pull-search-terms ---
@cli.command()
@_option_email()
@_option_cache()
@_option_file(help=("Provide a .csv file that contains an 'Search Terms' column. "))
def pull_search_terms(email, cache, file):
    """Retrieve the first result from each search term from Pubmed and store locally for further use."""
    df = pd.read_csv(file)
    if "Search Terms" not in df:
        raise ValueError("Provided .csv must contain an 'Search Terms' column.")
    results = sp.run_search(
        search_terms=df["Search Terms"].values, retmax=1, email=email
    )
    records = sp.load_results(results, email=email)
    sp.add_to_cache(records, just_warn=True, cache=cache)


# --- calculate-table ---
@cli.command()
@_option_cache()
@_option_file(
    help=(
        "A .csv file with the following columns: 'Species', 'Accessions', 'Human Host'. "
    )
)
@_option_rfc_file(
    required=False,
    help=(
        "Pickle file of a RandomForestClassifier containing columns to be used for training."
    ),
)
@_option_outfile(
    default="table.parquet.gzip",
    help=(
        "Output data table contains the following columns: 'Species', 'Human Host' and all calculated feature values. "
    ),
)
@_option_features_genomic()
@_option_features_gc()
@_option_features_kmers()
@_option_kmers_k()
def calculate_table(
    cache,
    file,
    rfc_file,
    outfile,
    features_genomic,
    features_gc,
    features_kmers,
    kmer_k,
):
    """Build a data table from given viral species and selected features."""
    df = pd.read_csv(file)
    rfc = None
    if rfc_file is not None:
        with open(rfc_file, "rb") as f:
            rfc = pickle.load(f)
    if set(["Species", "Accessions", "Human Host"]).issubset(df.columns) is False:
        raise ValueError(
            "Provided .csv file must contain 'Species', 'Accessions', and 'Human Host' columns."
        )
    if not features_genomic and not features_gc and not features_kmers:
        raise ValueError("No features selected.")
    sp.build_table(
        df,
        rfc=rfc,
        cache=cache,
        save=True,
        filename=outfile,
        genomic=features_genomic,
        gc=features_gc,
        kmers=features_kmers,
        kmer_k=kmer_k,
    )


# --- calculate-table-human ---
@cli.command()
@_option_cache()
@_option_rfc_file(
    required=False,
    help=("Pickle file of a RandomForestClassifier containing columns to be retained."),
)
@_option_outfile(
    default="table_human.parquet.gzip",
    help=(
        "Output data table contains calculated feature values for each genome in the cache. "
    ),
)
@_option_features_genomic()
@_option_features_gc()
@_option_features_kmers()
@_option_kmers_k()
def calculate_table_human(
    cache,
    rfc_file,
    outfile,
    features_genomic,
    features_gc,
    features_kmers,
    kmer_k,
):
    """Build a data table of the selected features from a given cache containing human genes."""
    rfc = None
    if rfc_file is not None:
        with open(rfc_file, "rb") as f:
            rfc = pickle.load(f)
    if not features_genomic and not features_gc and not features_kmers:
        raise ValueError("No features selected.")
    sp.build_table_human(
        rfc=rfc,
        cache=cache,
        save=True,
        filename=outfile,
        genomic=features_genomic,
        gc=features_gc,
        kmers=features_kmers,
        kmer_k=kmer_k,
    )


# --- cross-validation ---
@cli.command()
@_option_file(
    help=("Parquet file that contains the data table to run cross validation on.")
)
@_option_prefix(default="cv_")
@click.option(
    "--splits",
    "-s",
    default=5,
    show_default=True,
    help=("Number of folds to use for cross validation."),
)
def cross_validation(file, prefix, splits):
    """Run cross validation on the data table using default parameters"""
    X, y = sp.get_training_columns(table_filename=file)
    sp.cross_validation(X, y, plot=True, prefix=prefix, splits=splits)


# --- train ---
@cli.command()
@_option_file(help=("Parquet file that contains data table to use for training."))
@_option_outfile(
    default="rfc.p",
    help=("Pickle file of trained RandomForestClassifier."),
)
def train(file, outfile):
    """Train and save a RandomForestClassifier on the given data table using default parameters"""
    X, y = sp.get_training_columns(table_filename=file)
    sp.train_rfc(X, y, save=True, filename=outfile)


# --- predict ---
@cli.command()
@_option_file(help=("Parquet file that contains data table to use for prediction."))
@_option_rfc_file(
    help=("Pickle file of trained RandomForestClassifer to use for prediction.")
)
@_option_prefix()
def predict(file, rfc_file, prefix):
    X, y = sp.get_training_columns(table_filename=file)
    sp.predict_rfc(X, y, filename=rfc_file, plot=True, out_prefix=prefix)


# --- verify-cache ---
@cli.command()
@_option_cache()
def verify_cache(cache):
    """Check local cache for records that do not satisfy filters."""
    records = sp.load_from_cache(cache=cache)
    sp.filter_records(records, just_warn=True)


# --- plot-roc ---
@cli.command()
@click.argument(
    "files",
    nargs=-1,
)
@_option_outfile(
    default="roc_plot.png",
    help=("File name of generated plot."),
)
@click.option(
    "--title",
    "-t",
    default="ROC curve",
    show_default=True,
    help=("Title shown at the top of the plot."),
)
def plot_roc(files, outfile, title):
    """Plot ROC curve of passed files. CSV file should be formatted as the predict or cross-validation commands output"""
    sp.plot_roc(files, filename=outfile, title=title)
