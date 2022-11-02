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



def main():
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
    # quick % GC distribution check per organism
    # with box plot:
    fig_gc_dist_box, ax = plt.subplots()
    df.boxplot(ax=ax,
               by="Organism")
    ax.set_ylabel("% GC content in SARS-CoV-2 (and related animal) genome")
    fig_gc_dist_box.savefig("dataset_gc_percent_box_plot.png", dpi=300)





if __name__ == "__main__":
    main()
