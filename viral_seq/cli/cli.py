from time import perf_counter
import rich_click as click
import viral_seq.analysis.spillover_predict as sp
import pandas as pd
import pickle


# Function to allow defining click options that can be used for multiple commands
def shared_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


# Shared options
_option_email = [
    click.option(
        "--email",
        show_default=True,
        default="arhall@lanl.gov",
        help=(
            "Email used when communicating with Pubmed. "
            "You may be contacted if requests are deemed excessive. "
        ),
    )
]

_option_cache = [
    click.option(
        "--cache",
        "-c",
        default=".cache",
        show_default=True,
        help=("Specify local cache to use for this command. "),
    )
]


@click.group()
def cli():
    pass


# --- main ---
@cli.command()
@shared_options(_option_email)
@click.option(
    "--download_recs",
    type=int,
    show_default=True,
    default=0,
    help=(
        "Specify a number of sequence records "
        "to download (per search category), either generating "
        "or expanding the local cache with records it doesn't "
        "already have."
    ),
)
@click.option(
    "--save_model",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Serialize (pickle) the trained ML model(s) to"
        " disk so that they can be packaged/reused elsewhere"
    ),
)
@click.option(
    "--load_model",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Load the trained (serialized/pickled) ML model(s) in to"
        " memory so that they can be used immediately in cases where"
        " it is not necessary to iterate on the model proper. This will"
        " require that `save_model` has been used locally first."
    ),
)
def main(download_recs, email, save_model, load_model):
    start_sec = perf_counter()
    sp.main(
        download_recs=download_recs,
        email=email,
        save_model=save_model,
        load_model=load_model,
    )
    end_sec = perf_counter()
    execution_time_sec = end_sec - start_sec
    print(f"viral_seq execution time (s): {execution_time_sec:.2f}")


# --- search-data ---
@cli.command()
@shared_options(_option_email)
@shared_options(_option_cache)
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
    results = sp.run_search(search_terms=query, retmax=retmax, email=email)
    records = sp.load_results(results, email=email)
    sp.add_to_cache(records, cache=cache)


# --- pull-data ---
@cli.command()
@shared_options(_option_email)
@shared_options(_option_cache)
@click.option(
    "--file",
    "-f",
    required=True,
    help=("Provide a .csv file that contains an 'Accessions' column. "),
)
def pull_data(email, cache, load):
    """Retrieve accessions from Pubmed and store locally for further use."""
    df = pd.read_csv(load)
    assert "Accessions" in df
    # Entries in 'Accessions' column may be space delimited accessions for species with multiple segments
    accessions = set((" ".join(df["Accessions"].values)).split())
    records = sp.load_results(accessions, email=email)
    sp.add_to_cache(records, cache=cache)


# --- calculate-table ---
@cli.command()
@shared_options(_option_cache)
@click.option(
    "--file",
    "-f",
    required=True,
    help=(
        "A .csv file with the following columns: 'Species', 'Accessions', 'Human Host'. "
    ),
)
@click.option(
    "--rfc-file",
    default=None,
    help=(
        "Pickle file of a RandomForestClassifier containing columns to be used for training."
    ),
)
@click.option(
    "--outfile",
    "-o",
    default="table.parquet.gzip",
    show_default=True,
    help=(
        "Output data table contains the following columns: 'Species', 'Human Host' and all calculated feature values. "
    ),
)
@click.option(
    "--features-genomic",
    "-g",
    is_flag=True,
    help=("Calculate viral genomic features, including: ..."),
)
@click.option(
    "--features-gc", "-gc", is_flag=True, help=("Calculate GC content feature. ")
)
@click.option(
    "--features-kmers",
    "-kmers",
    is_flag=True,
    help=("Calculate amino acid Kmer features."),
)
@click.option(
    "--kmer-k",
    "-k",
    default=10,
    show_default=True,
    help=("K value to use for amino acid Kmer calculation."),
)
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
    assert set(["Species", "Accessions", "Human Host"]).issubset(df.columns)
    assert features_genomic or features_gc or features_kmers
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


# --- cross-validation ---
@cli.command()
@click.option(
    "--file",
    "-f",
    required=True,
    help=("Parquet file that contains the data table to run cross validation on."),
)
@click.option(
    "--prefix",
    "-p",
    default="cv_",
    show_default=True,
    help=("Prefix appended to output files."),
)
def cross_validation(file, prefix):
    """Run 5 fold cross validation on the data table using default parameters"""
    X, y = sp.get_training_columns(table_filename=file)
    sp.cross_validation(X, y, plot=True, prefix=prefix)


# --- train ---
@cli.command()
@click.option(
    "--file",
    "-f",
    required=True,
    help=("Parquet file that contains data table to use for training."),
)
@click.option(
    "--outfile",
    "-o",
    default="rfc.p",
    show_default=True,
    help=("Pickle file of trained RandomForestClassifier."),
)
def train(file, outfile):
    """Train and save a RandomForestClassifier on the given data table using default parameters"""
    X, y = sp.get_training_columns(table_filename=file)
    sp.train_rfc(X, y, save=True, filename=outfile)


# --- predict ---
@cli.command()
@click.option(
    "--file",
    "-f",
    required=True,
    help=("Parquet file that contains data table to use for prediction."),
)
@click.option(
    "--rfc-file",
    required=True,
    help=("Pickle file of trained RandomForestClassifer to use for prediction."),
)
@click.option(
    "--out-prefix",
    "-o",
    default="cli_",
    show_default=True,
    help=("Prefix appended to output files."),
)
def predict(file, rfc_file, out_prefix):
    X, y = sp.get_training_columns(table_filename=file)
    sp.predict_rfc(X, y, filename=rfc_file, plot=True, out_prefix=out_prefix)


# --- verify-cache ---
@cli.command()
@shared_options(_option_cache)
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
@click.option(
    "--outfile",
    "-o",
    default="roc_plot.png",
    show_default=True,
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
