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



def main():
    rng = default_rng(676906)
    # let's start off by scraping the pubmed
    # nucleotide database from genomic SARS-CoV-2 RNA

    # restrict retrieved sequences
    # during early development
    retmax = 5000
    print(f"Total records requested in search: {retmax}")

    # we only scrape if we don't have the local data
    # already though:
    cache_path = Path(".cache/seqs.p")
    if Path(".cache/seqs.p").exists():
        print("Retrieving SARS-CoV-2 genomic RNA data from local cache.")
        with open(cache_path, "rb") as records_cache:
            records = pickle.load(records_cache)
    else:
        print("Retrieving SARS-CoV-2 genomic RNA remotely from Pubmed.")

        # let Pubmed team email us if they find
        # that our scraping is too aggressive
        # (rather than ban us outright)
        Entrez.email = "treddy@lanl.gov"

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
        for search_term in ["SARS-CoV-2 genomic RNA",
                            # 96.1% nucleotide similarity with SARS-CoV-2
                            # for RatG13, but it has S mutations that suggest
                            # it may not enter via ACE2 (also lacks furin cleavage site)
                            "Bat coronavirus complete genome"]:
            handle = Entrez.esearch(db="nucleotide",
                                    term=search_term,
                                    retmax=retmax,
                                    idtype="acc")
            record = Entrez.read(handle)
            handle.close()
            print(f"total {search_term} sequences found on Pubmed:", record["Count"])
            print(f"total {search_term} sequences *retrieved* from Pubmed:", record["RetMax"])
            assert len(record["IdList"]) <= retmax
            # retrieve the sequence data
            handle = Entrez.efetch(db="nuccore", id=record["IdList"], rettype="gb", retmode="text")
            records += list(SeqIO.parse(handle, "gb"))

        # cache the sequence data so we don't overwhelm Pubmed
        # and get banned when iterating on the code
        p = Path(".cache/")
        p.mkdir()
        cache_path = p / "seqs.p"
        print("Serializing Pubmed SARS-CoV-2 records into pickle file.")
        with cache_path.open("wb") as records_cache:
            pickle.dump(records, records_cache)


    # sanity check -- based on this manuscript:
    # https://doi.org/10.1038/s41598-020-69342-y
    # there should be > 29,000 nucleotides in the
    # SARS-CoV-2 genome
    # NOTE: relaxing the checks as I start incorporating viral
    # sequences from other organisms, like bats
    print("Sanity checking/filtering retrieved SARS-CoV-2 (and related) RNA genome sizes:")
    assert len(records) <= retmax * 2
    retained_records = []
    for record in tqdm(records):
        # each record is a Bio.SeqRecord.SeqRecord
        actual_len = len(record)
        if record.description.startswith("Homo sapiens") or record.description.endswith("mRNA") or ("partial"
           in record.description):
            # likely "data pollution"/bad sequences
            # picked up in search
            continue
        assert actual_len > 26000, f"Actual sequence length is {actual_len}"
        assert actual_len < 31000, f"Actual sequence length is {actual_len}"
        retained_records.append(record)

    filtered_rec_count = len(retained_records)
    print("Total records retained after filtering:", filtered_rec_count)


    # next, try to plot the % GC content for the SARS-CoV-2
    # RNA genomes currently getting pulled in
    gc_content_data = np.empty(shape=(filtered_rec_count,), dtype=np.float64)
    for idx, record in enumerate(retained_records):
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
    for record in retained_records:
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
                                            legend=False)
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




if __name__ == "__main__":
    main()
