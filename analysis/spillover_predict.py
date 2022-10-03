import pickle
from pathlib import Path

from tqdm import tqdm
from Bio import Entrez, SeqIO



def main():
    # let's start off by scraping the pubmed
    # nucleotide database from genomic SARS-CoV-2 RNA

    # restrict to 100 retrieved sequences
    # during early development
    retmax = 100

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

        handle = Entrez.esearch(db="nucleotide",
                                term="SARS-CoV-2 genomic RNA",
                                retmax=retmax,
                                idtype="acc")
        record = Entrez.read(handle)
        handle.close()
        # a bit over 64,000 at the time of writing:
        print("total SARS-CoV-2 genomic RNA sequences found on Pubmed:", record["Count"])
        # set to around 100 actually retrieved for now:
        print("total SARS-CoV-2 genomic RNA sequences *retrieved* from Pubmed:", record["RetMax"])
        assert len(record["IdList"]) == retmax
        # retrieve the sequence data
        handle = Entrez.efetch(db="nuccore", id=record["IdList"], rettype="gb", retmode="text")
        records = list(SeqIO.parse(handle, "gb"))

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
    print("Sanity checking retrieved SARS-CoV-2 RNA genome sizes:")
    assert len(records) == retmax
    for record in tqdm(records):
        # each record is a Bio.SeqRecord.SeqRecord
        assert len(record) > 29000
        assert len(record) < 30000


if __name__ == "__main__":
    main()
