"""
Provide some parsing utility functions and data
sanity checks related to the dataset described
in https://re-git.lanl.gov/treddy/ldrd_virus_work/-/issues/27.

This should be the data used for Figure 3 in:

Mollentze N, Babayan SA, Streicker DG (2021) Identifying and prioritizing potential human-infecting viruses from their genome sequences. PLOS Biology 19(9): e3001390. https://doi.org/10.1371/journal.pbio.3001390

But it appears to actually be an "extended" version of that dataset
with more samples added vs. the published version (see below...)

CAUTION: the repo this data is pulled from is GPL-3 licensed:
https://github.com/Nardus/zoonotic_rank

The idea is to first parse/sanity check this data set in isolation
from the main control flow of our analysis, but to build up any
relevant infrastructure for loading/double checking the data
in reusable functions that the main control flow can use later.
"""

import time
from tqdm import tqdm
import pandas as pd
from Bio import Entrez, SeqIO


if __name__ == "__main__":
    start = time.perf_counter()
    # this appears to be an "extended" version of the dataset
    # used to produce Figure 3, so we will likely have to use
    # their R code to re-analyze/compare?
    df = pd.read_csv("NovelVirus_Hosts_Curated.csv")
    print("df:\n", df)
    df.info()
    # we'd like to confirm that there are indeed 758 unique viral species
    # present, as a sanity check that we have the exact data referred to in
    # the manuscript

    # so, I think we need to scrape the viral species data in
    # and then parse it, since the species data in the deposited CSV
    # appears to focus on the host only


    # on page 7/25 of the manuscript, the authors describe the human-origin
    # samples of the dataset as having the following zoonotic potentials:
    # very high: N = 36
    # high: N = 44
    # medium: N = 30
    # low: N = 3
    # however, when I check this "extended" version of the dataset, it has clearly
    # grown to include more human samples, so we'll have to repeat the analysis
    # using their GitHub code perhaps?
    expected_human_origin_samples = 36 + 44 + 30 + 3
    actual_human_origin_samples = (df["host_corrected"] == "Homo sapiens").sum()
    msg = f"Samples from humans (N={actual_human_origin_samples}) does not match description in manuscript (N={expected_human_origin_samples})"
    assert actual_human_origin_samples == expected_human_origin_samples, msg

    """
    Entrez.email = "treddy@lanl.gov"
    accessions = df["accession"]
    viral_families = set()
    missing_taxonomies = 0
    print("retrieving records for Mollentze Figure 3 data")
    handle = Entrez.efetch(db="nuccore",
                           rettype="gb",
                           id=accessions,
                           idtype="acc",
                           retmode="text")
    print("parsing records for Mollentze Figure 3 data")
    records = list(SeqIO.parse(handle, "gb"))
    for record in tqdm(records):
        family_found = 0
        taxonomy = record.annotations["taxonomy"]
        for entry in taxonomy:
            if "viridae" in entry:
                viral_families.add(entry)
                family_found += 1
                break
        if len(taxonomy) == 0 or family_found == 0:
            missing_taxonomies += 1
        # we're really just trying to assess the claim of 758
        # unique species in the paper, so as long as the taxonomy
        # has 758 unique entries of some sort, we'll probably
        # be happy without additional hand curation here
    print("Viral families:", viral_families)
    print("Estimate of number of unique viral families:", len(viral_families))
    print("Num records with missing taxonomies:", missing_taxonomies)
    handle.close()
    end = time.perf_counter()
    elapsed = end - start
    print(f"analysis complete in {elapsed:.1f} seconds")
    """
