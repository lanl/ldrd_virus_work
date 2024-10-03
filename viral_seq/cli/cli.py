import rich_click as click
import viral_seq.analysis.spillover_predict as sp
import viral_seq.analysis.get_features as gf
import pandas as pd
import pickle
import os


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
    search_terms = query.split()
    results = sp.run_search(search_terms=search_terms, retmax=retmax, email=email)
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
@click.option(
    "--no-filter",
    "-n",
    is_flag=True,
    help=(
        "Will warn about triggered data filters, but will still add all results to cache. "
    ),
)
def pull_data(email, cache, file, no_filter):
    """Retrieve accessions from Pubmed and store locally for further use."""
    df = pd.read_csv(file)
    if "Accessions" not in df:
        raise ValueError("Provided .csv must contain an Accessions column.")
    # Entries in 'Accessions' column may be space delimited accessions for species with multiple segments
    accessions = set((" ".join(df["Accessions"].values)).split())
    records = sp.load_results(accessions, email=email)
    sp.add_to_cache(records, cache=cache, just_warn=no_filter)


# --- pull-ensembl-transcripts
@cli.command()
@shared_options(_option_email)
@shared_options(_option_cache)
@click.option(
    "--file",
    "-f",
    required=True,
    help=("A .txt file of white space delimited ensemble transcript ids"),
)
def pull_ensembl_transcripts(email, cache, file):
    """Retrieve the refseq results from Pubmed for the provided transcripts and store locally for further use."""
    with open(file, "r") as f:
        lines = f.readlines()
    search_term = "(biomol_mrna[PROP] AND refseq[filter]) AND("
    transcripts = lines[0].split()
    first = True
    for transcript in transcripts:
        if first:
            first = False
        else:
            search_term += "OR "
        search_term += transcript + "[All Fields] "
    search_term += ")"
    results = sp.run_search(
        search_terms=[search_term], retmax=len(transcripts), email=email
    )
    records = sp.load_results(results, email=email)
    sp.add_to_cache(records, just_warn=True, cache=cache)


# --- calculate-table ---
@cli.command()
@shared_options(_option_cache)
@click.option(
    "--file",
    "-f",
    required=True,
    help=(
        "A .csv file with the following columns: 'Species', 'Accessions', and a target column with name specified by '--target-column'. "
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
        "Output data table contains the following columns: An unnamed index column, 'Species', the target column, and all calculated feature values. "
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
    default="10",
    show_default=True,
    help=(
        "K value to use for amino acid Kmer calculation. For multiple, enter a whitespace delimited list."
    ),
)
@click.option(
    "--features-kmers-pc",
    "-kmerspc",
    is_flag=True,
    help=("Calculate amino acid PC-Kmer features."),
)
@click.option(
    "--kmer-k-pc",
    "-kpc",
    default="10",
    show_default=True,
    help=(
        "K value to use for amino acid PC-Kmer calculation. For multiple, enter a whitespace delimited list."
    ),
)
@click.option(
    "--similarity-genomic",
    "-sg",
    is_flag=True,
    help=("Calculate genomic similarity features"),
)
@click.option(
    "--similarity-cache",
    "-sc",
    help=(
        "Cache folders to use when calculating similarity features, whitespace delimited. Required if similarity feature is selected."
    ),
)
@click.option(
    "--uni-select",
    "-u",
    is_flag=True,
    help=("Filter features after calculation based on univariate feature selection."),
)
@click.option(
    "--uni-type",
    "-ut",
    default="mutual_info_classif",
    show_default=True,
    help=(
        "Selection method for univariate feature selection. Options: chi2, mutual_info_classif, f_classif"
    ),
)
@click.option(
    "--num-select",
    "-n",
    default=1_000,
    show_default=True,
    help=(
        "Number of features per category to retain when filtering with univariate selection."
    ),
)
@click.option(
    "--random-state",
    "-r",
    default=123456789,
    help=("Random seed value used for mutual_info_classif. Does nothing otherwise."),
)
@click.option(
    "--target-column",
    "-tc",
    default="Human Host",
    help=("Target column to be used for binary classification."),
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
    features_kmers_pc,
    kmer_k_pc,
    similarity_genomic,
    similarity_cache,
    uni_select,
    uni_type,
    num_select,
    random_state,
    target_column,
):
    """Build a data table from given viral species and selected features."""
    df = pd.read_csv(file)
    rfc = None
    if rfc_file is not None:
        with open(rfc_file, "rb") as f:
            rfc = pickle.load(f)

    needed_columns = ["Species", "Accessions", target_column]
    if set(needed_columns).issubset(df.columns) is False:
        raise ValueError(
            "Provided .csv file must contain 'Species', 'Accessions', and '"
            + str(target_column)
            + "' columns."
        )

    df = df[needed_columns]

    if (
        not features_genomic
        and not features_gc
        and not features_kmers
        and not features_kmers_pc
    ):
        raise ValueError("No features selected.")
    if similarity_genomic and not features_genomic:
        raise ValueError(
            "To calculate genomic similarity features, you must calculate genomic features."
        )
    if similarity_genomic and not similarity_cache:
        raise ValueError(
            "To calculate similarity features, you must provide at least one cache."
        )
    if features_kmers:
        kmer_k = [int(i) for i in kmer_k.split()]
    if features_kmers_pc:
        kmer_k_pc = [int(i) for i in kmer_k_pc.split()]
    df_feats = sp.build_table(
        df,
        rfc=rfc,
        cache=cache,
        kmers_pc=features_kmers_pc,
        kmer_k_pc=kmer_k_pc,
        save=(not similarity_genomic),
        filename=outfile,
        genomic=features_genomic,
        gc=features_gc,
        kmers=features_kmers,
        kmer_k=kmer_k,
        uni_select=uni_select,
        uni_type=uni_type,
        num_select=num_select,
        random_state=random_state,
        target_column=target_column,
    )
    if similarity_genomic:
        for sim_cache in similarity_cache.split():
            this_table = sp.build_table(
                cache=sim_cache,
                save=False,
                genomic=similarity_genomic,
                gc=False,
                kmers=False,
                kmers_pc=False,
                target_column=target_column,
            )
            df_feats = gf.get_similarity_features(
                this_table, df_feats, suffix=os.path.basename(sim_cache)
            )
        sp.save_files(df_feats, outfile)


# --- cross-validation ---
@cli.command()
@click.argument(
    "files",
    nargs=-1,
)
@click.option(
    "--prefix",
    "-p",
    default="cv_",
    show_default=True,
    help=("Prefix prepended to output filenames."),
)
@click.option(
    "--splits",
    "-s",
    default=5,
    show_default=True,
    help=("Number of folds to use for cross validation."),
)
def cross_validation(files, prefix, splits):
    """Run cross validation on the data table using default parameters"""
    X, y = sp.get_training_columns(table_filename=files)
    sp.cross_validation(X, y, plot=True, prefix=prefix, splits=splits)


# --- train ---
@cli.command()
@click.argument(
    "files",
    nargs=-1,
)
@click.option(
    "--outfile",
    "-o",
    default="rfc.p",
    show_default=True,
    help=("Pickle file of trained RandomForestClassifier."),
)
def train(files, outfile):
    """Train and save a RandomForestClassifier on the given data table using default parameters"""
    X, y = sp.get_training_columns(table_filename=files)
    sp.train_rfc(X, y, save=True, filename=outfile)


# --- predict ---
@cli.command()
@click.argument(
    "files",
    nargs=-1,
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
    help=("Prefix prepended to output files."),
)
def predict(files, rfc_file, out_prefix):
    X, y = sp.get_training_columns(table_filename=files)
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
