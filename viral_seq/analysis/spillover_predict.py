import pickle
from pathlib import Path

from tqdm import tqdm
from Bio import Entrez, SeqIO
from Bio.SeqUtils import GC
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from numpy.random import default_rng



def main(download_recs: int):
    """
    Parameters
    ----------
    download_recs : int
        The number of records (genetic sequences) to download per
        search category. At the time of writing there are
        two search categories (human, bat), so a value of ``100``
        would attempt to retrive ``100`` records per search, though
        less may be found and in theory a single record could contain
        more than one genetic sequence.
    """
    rng = default_rng(676906)
    # let's start off by scraping the pubmed
    # nucleotide database from genomic SARS-CoV-2 RNA

    retmax = download_recs
    print(f"Total records requested in search (per search term): {retmax}")

    # we only scrape from the online database if requested,
    # and even if we requested we should only add sequences
    # if they correspond to NCBI accession numbers we do not
    # yet have (doing this periodically on a given machine/workspace
    # probably makes sense, since NCBI sequence counts double every
    # ~18 months and the local sequence cache may become outdated quickly)

    cache_path = Path(".cache")
    # envsioned local sequence cache layout:
    # .cache/ folder
    # subfolders are formatted as accession numbers
    # and each subfolder should contain the FASTA and GENBANK
    # sequences for that accession number

    # if you try to run the analysis with no cache,
    # and no request to fill the cache with new records,
    # that's an error
    if not cache_path.exists() and not download_recs:
        raise ValueError("No local sequence cache exists and no request "
                         "to populate the cache with new sequences was made.")

    # if the ".cache" folder exists but there are no sequence
    # records, that's also an error
    local_records = []
    # TODO: properly check that there are valid records present, rather
    # than just any subfolders
    if not cache_path.exists():
        cache_path.mkdir(parents=False, exist_ok=False)
    cache_subdirectories = [x for x in cache_path.iterdir() if x.is_dir()]
    if not cache_subdirectories and not download_recs:
        raise ValueError("No request to populate the local sequence cache "
                         "was made and there are no local accession records "
                         "available.")

    if download_recs > 0:
        print("Retrieving sequence data remotely from Pubmed.")

        # let Pubmed team email us if they find
        # that our scraping is too aggressive
        # (rather than ban us outright)
        Entrez.email = "treddy@lanl.gov"

        # through empirical testing, I determined that we can't really efficiently
        # exclude the initial Entrez search from containing records/accession numbers
        # we already have in the local cache, because the string of `NOT <accession no.>`
        # elements explodes to a size the server can't handle, resulting in HTTP 500
        # errors

        # based on feedback from NCBI support, it sounds like there are two viable options
        # 1) the most efficient is probably to search for new records based on the date they
        # were updated, so that we only pull in new records we don't have locally--however,
        # when starting with an empty or small initial local cache, this doesn't seem very 
        # helpful/practical for building up progressively (we want all the dates at first...)
        # 2) Instead of excluding by accession number on the initial search, we could perform
        # the same search each time, but filter on accession number AFTER the initial search
        # and BEFORE the actual *retrieval* of records

        # try to use approach 2) below...

        set_local_accessions = set()
        if cache_path.exists():
            set_local_accessions = set([p.name for p in cache_path.iterdir()])

        # this manuscript:
        # https://doi.org/10.1126/science.abm1208
        # Obermeyer et al., Science 376, 1327–1332 (2022)
        # cites more than 6 million SARS-CoV-2 genomes
        # available, but let's start with the smaller grab
        # from Pubmed initially at least...

        # in order to get multispecies data we combine searches
        # for SARS-CoV-2 with searches for closely related
        # viruses in bats and pangolins (just bats for now..)
        # see: Nature volume 604, pages 330–336 (2022)
        records = []
        for search_term in ['("Severe acute respiratory syndrome coronavirus 2"[Organism] OR sars-cov-2[All Fields]) AND genome[All Fields]',
                            # 96.1% nucleotide similarity with SARS-CoV-2
                            # for RatG13, but it has S mutations that suggest
                            # it may not enter via ACE2 (also lacks furin cleavage site)
                            "Bat coronavirus complete genome"]:

            # try to prevent timeouts on large search queries
            # by batching
            search_batch_size = min(10_000, retmax)
            acc_set = set()
            count = 0
            remaining = retmax
            print(f"starting Entrez esearch (search batch size is: {search_batch_size})")
            for retstart in tqdm(range(1, retmax + 1, search_batch_size)):
                if remaining < search_batch_size:
                    actual_retmax = remaining
                else:
                    actual_retmax = search_batch_size
                handle = Entrez.esearch(db="nucleotide",
                                        term=search_term,
                                        retstart=retstart,
                                        retmax=actual_retmax,
                                        idtype="acc",
                                        usehistory="y")
                search_results = Entrez.read(handle)
                acc_set = acc_set.union(search_results["IdList"])
                count += int(search_results["Count"])
                remaining -= search_batch_size
                handle.close()
            print("after Entrez esearch")

            print("filtering accession records already present locally")
            new_acc_set = acc_set.difference(set_local_accessions)

            print("number of new records from online search to consider for inclusion in local cache:", len(new_acc_set))
            if len(new_acc_set) == 0:
                continue

            # these are too verbose now, maybe use logging DEBUG eventually
            #print(f"total {search_term} sequences found on Pubmed:", count)
            #print(f"total {search_term} sequences *retrieved* from Pubmed:", search_results["RetMax"])
            assert len(acc_set) <= retmax
            # retrieve the sequence data
            batch_size = min(retmax, 100, len(new_acc_set))
            numrecs = min(retmax, count, len(new_acc_set))
            print(f"fetching {numrecs} Entrez records with batch_size {batch_size}:")
            new_acc_list = list(new_acc_set)
            for start in tqdm(range(0, numrecs, batch_size)):
                handle = Entrez.efetch(db="nuccore",
                                       rettype="gb",
                                       id=new_acc_list[start:start + batch_size],
                                       idtype="acc",
                                       retmode="text")
                records += list(SeqIO.parse(handle, "gb"))
                handle.close()

        # with the records list populated from the online
        # search, we next want to grow the local cache with
        # folders + FASTA/GENBANK format files that are not
        # already there
        print("performing quality analysis of sequences and adding to local cache if they pass...")
        num_recs_added = 0
        num_recs_excluded = 0
        for record in tqdm(records):
            # we also sanity check each record before
            # trying to cache it locally;
            # each record is a Bio.SeqRecord.SeqRecord
            actual_len = len(record)
            if record.description.startswith("Homo sapiens") or record.description.endswith("mRNA") or ("partial"
               in record.description):
                # likely "data pollution"/bad sequences
                # picked up in search
                num_recs_excluded += 1
                continue
            # these aren't assertions yet at this point because
            # we don't want to interrupt the population of the local
            # cache because of a few bad apples from an online search
            if actual_len < 26000 or actual_len > 31000:
                num_recs_excluded += 1
                continue
            if ("DNA" in record.annotations["molecule_type"] or 
                "Klebsiella" in record.annotations["organism"]):
                # for now, exclude DNA and focus on RNA coronaviruses;
                # this should help prevent the sequence that got pulled in
                # for: https://re-git.lanl.gov/treddy/ldrd_virus_work/-/issues/13
                # TODO: if we expand to include i.e., DNA viral genomes, we'll
                # almost certainly need more sophisticated filtering than just
                # one bacterial species though
                num_recs_excluded += 1
                continue
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

        print(f"number of records added to the local cache from online search: {num_recs_added}")
        print(f"number of records excluded from the online search because of QA: {num_recs_excluded}")
        num_recs = len(records)
        max_recs = retmax * 2 # 2 search terms (human + bat)
        assert num_recs <= max_recs, f"num_recs={num_recs} is larger than max expected of {max_recs}"


    # whether download_recs > 0 or not, we will always
    # consider the FASTA/GENBANK records stored in .cache
    # as the source of genetic sequence data that we we are
    # working with, which means doing some I/O now to pull
    # that data in from disk

    # some of the sequence-length/related sanity checking
    # also should be guaranteed complete by now since we
    # should have filtered the records before writing to disk;
    # nonetheless, reapplying some sanity checks after/while
    # reading the data back in probably makes sense (check for
    # data/cache corruption, etc.)

    print("populating cache_subdirectories object from the local cache")
    cache_subdirectories = []
    for entry in cache_path.iterdir():
        if entry.is_dir():
            cache_subdirectories.append(entry)
    print("total number of records (sequence folders) in updated local cache:", len(cache_subdirectories))


    print("loading the local records cache (genetic sequences) into memory")
    records = []
    for record_folder in tqdm(cache_subdirectories):
        # genbank format has more metadata we can use so
        # focus on that for now; eventually we might assert
        # that the FASTA file has the same sequence read in
        # as sanity check
        genbank_file = list(record_folder.glob("*.genbank"))[0]
        with open(genbank_file) as genfile:
            record = list(SeqIO.parse(genfile, "genbank"))[0]
            records.append(record)

    # sanity check -- based on this manuscript:
    # https://doi.org/10.1038/s41598-020-69342-y
    # there should be > 29,000 nucleotides in the
    # SARS-CoV-2 genome
    # NOTE: relaxing the checks as I start incorporating viral
    # sequences from other organisms, like bats
    print("Sanity checking/filtering SARS-CoV-2 (and related) RNA genome records retrieved from the local cache:")
    for record in tqdm(records):
        # each record is a Bio.SeqRecord.SeqRecord
        actual_len = len(record)
        # likely "data pollution"/bad sequences
        # picked up in search
        assert not (record.description.startswith("Homo sapiens") or record.description.endswith("mRNA") or ("partial"
           in record.description))
        assert actual_len > 26000, f"Actual sequence length is {actual_len}"
        assert actual_len < 31000, f"Actual sequence length is {actual_len}"

    filtered_rec_count = len(records)
    print("Total records retained:", filtered_rec_count)

    # next, try to plot the % GC content for the SARS-CoV-2
    # RNA genomes currently getting pulled in
    gc_content_data = np.empty(shape=(filtered_rec_count,), dtype=np.float64)
    for idx, record in enumerate(records):
        gc_content_data[idx] = GC(record.seq)

    fig_gc_dist = plt.figure()
    ax = fig_gc_dist.add_subplot(111)
    ax.set_xlim(30, 50)
    ax.set_xlabel("% GC content in SARS-CoV-2 RNA genome")
    ax.set_ylabel("Frequency")
    ax.set_title(f"% GC content histogram for SARS-CoV-2 RNA Genomes (N={filtered_rec_count})")
    ax.hist(gc_content_data,
            bins=int(filtered_rec_count/10),
            )
    # human genomic %GC content from
    # Piovesan et al. (2019) BMC Research Notes
    ax.vlines(x=40.9,
              ymin=0,
              ymax=2000,
              ls="--",
              color="green")
    # bat genomic %GC content from
    # Kasai et al. (2013) Chromosoma
    ax.vlines(x=42.3,
              ymin=0,
              ymax=2000,
              ls="--",
              color="black")
    # Arabidopsis thaliana % GC content from
    # Michael et al. (2018)
    ax.vlines(x=36,
              ymin=0,
              ymax=2000,
              ls="--",
              color="orange")
    fig_gc_dist.savefig("percent_gc_content_histogram.png",
                dpi=300)

    # let's construct a pandas dataframe, since this is
    # a reasonably convenient data structure for interacting
    # with i.e., scikit-learn and also for general analysis/inspection/visualization

    data_dict = {}
    for record in records:
        if "human" in record.description.lower():
            organism_name = "human"
        elif "bat" in record.description.lower():
            organism_name = "bat"
        else:
            organism_name = "unknown"

        data_dict[record.id] = [GC(record.seq), organism_name]

    df = pd.DataFrame.from_dict(data_dict,
                                orient="index",
                                columns=["Genome %GC", "Organism"])
    df.columns.name = "Genome ID"
    print("-" * 30)
    print("df:\n", df)
    print("-" * 30)
    df.info()
    print("-" * 30)
    print(df.describe())
    print("-" * 30)
    # check number of genomes per organism
    # in the dataset (DataFrame)
    print("Breakdown of sample size (num genomes) per organism in SARS-CoV-2 (and related) dataset:\n", df.value_counts("Organism"))
    fig_sample_sizes, ax = plt.subplots()
    df.groupby("Organism").count().plot.bar(ax=ax,
                                            rot=0,
                                            legend=False,
                                            log=True)
    ax.set_ylabel("Sample Size (number of genomes)")
    ax.set_title("Initial SARS-CoV-2 (and related animal) genome dataset")
    fig_sample_sizes.savefig("sample_sizes.png", dpi=300)
    # quick % GC distribution check per organism
    # with box plot:
    fig_gc_dist_box, ax = plt.subplots()
    df.boxplot(ax=ax,
               by="Organism")
    ax.set_ylabel("% GC content in SARS-CoV-2 (and related animal) genome")
    fig_gc_dist_box.savefig("dataset_gc_percent_box_plot.png", dpi=300)

    # early/crude work with a random forest classifier, just to
    # get a stats/ML workflow going...

    print("Training a Random Forest Classifier on %GC content of viral genome")
    # we temporarily reshape our data because it currently
    # only has a single feature (GC content)
    X = df["Genome %GC"].to_numpy().reshape(-1, 1)
    Y = df["Organism"]
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, Y)

    # let's perform 5-fold cross validation to get
    # a sense for the estimator accuracy
    fig_cross_val, (ax_macro, ax_micro) = plt.subplots(2, 1)
    scores_macro = cross_val_score(clf, X, Y.values, cv=5, scoring="f1_macro")
    print(f"Random Forest Cross Validation F1 macro has an average of {scores_macro.mean()} and std dev of {scores_macro.std()}:")
    scores_micro = cross_val_score(clf, X, Y.values, cv=5, scoring="f1_micro")
    print(f"Random Forest Cross Validation F1 micro has an average of {scores_micro.mean()} and std dev of {scores_micro.std()}:")

    # the cross-validation accuracy should be much lower
    # if we shuffle the features relative to the labels
    X_shuffled = X.copy()
    rng.shuffle(X_shuffled)
    shuffle_scores_macro = cross_val_score(clf, X_shuffled, Y.values, cv=5, scoring="f1_macro")
    shuffle_scores_micro = cross_val_score(clf, X_shuffled, Y.values, cv=5, scoring="f1_micro")
    print(f"(Randomly shuffled features, relative to labels) Random Forest Cross Validation F1 macro has an average of {shuffle_scores_macro.mean()} and std dev of {shuffle_scores_macro.std()}:")
    print(f"(Randomly shuffled features, relative to labels) Random Forest Cross Validation F1 micro has an average of {shuffle_scores_micro.mean()} and std dev of {shuffle_scores_micro.std()}:")

    # perhaps even worse would be a random set
    # of % GC values?
    X_random = rng.random(X.shape)
    random_scores_macro = cross_val_score(clf, X_random, Y.values, cv=5, scoring="f1_macro")
    random_scores_micro = cross_val_score(clf, X_random, Y.values, cv=5, scoring="f1_micro")
    print(f"(Randomly generated % GC content) Random Forest Cross Validation F1 macro has an average of {random_scores_macro.mean()} and std dev of {random_scores_macro.std()}:")
    print(f"(Randomly generated % GC content) Random Forest Cross Validation F1 micro has an average of {random_scores_micro.mean()} and std dev of {random_scores_micro.std()}:")

    score_means_macro = [scores_macro.mean(), shuffle_scores_macro.mean(), random_scores_macro.mean()]
    score_std_macro = [scores_macro.std(), shuffle_scores_macro.std(), random_scores_macro.std()]
    x_lab = ["original data", "shuffled GC content", "randomized GC content"]

    score_means_micro = [scores_micro.mean(), shuffle_scores_micro.mean(), random_scores_micro.mean()]
    score_std_micro = [scores_micro.std(), shuffle_scores_micro.std(), random_scores_micro.std()]

    ax_macro.bar(height=score_means_macro,
           yerr=score_std_macro,
           capsize=20.0,
           x=x_lab)
    ax_macro.set_title("5-fold cross validation of random forest")
    ax_macro.set_ylabel("F1 macro of\n classification by % GC")
    ax_micro.bar(height=score_means_micro,
           yerr=score_std_micro,
           capsize=20.0,
           x=x_lab)
    ax_micro.set_ylabel("F1 micro of\n classification by % GC")
    for ax in [ax_micro, ax_macro]:
        ax.set_ylim(0, 1)
    fig_cross_val.savefig("cross_vali_rand_forest.png", dpi=300)
