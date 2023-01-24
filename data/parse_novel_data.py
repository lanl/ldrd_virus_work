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
from Bio.SeqUtils import GC
import numpy as np


if __name__ == "__main__":
    start = time.perf_counter()
    # this appears to be an "extended" version of the dataset
    # used to produce Figure 3, so we will likely have to use
    # their R code to re-analyze/compare?
    df = pd.read_csv("NovelVirus_Hosts_Curated.csv")
    df["accession"] = df["accession"].astype("string")
    print("df:\n", df)
    df.info()
    # we'd like to confirm that there are indeed 758 unique viral species
    # present, as a sanity check that we have the exact data referred to in
    # the manuscript

    # so, I think we need to scrape the viral species data in
    # and then parse it, since the species data in the deposited CSV
    # appears to focus on the host only

    # from personal correspondence with Nardus (see:
    # https://re-git.lanl.gov/treddy/ldrd_virus_work/-/issues/27#note_506903)
    # some duplicate data was filtered using an additional file he emailed to me,
    # and is based on the code here:
    # https://github.com/Nardus/zoonotic_rank/blob/42f15a07ffdfc1ba425741233009f9c61bb3bf48/Scripts/Plotting/MakeFigure3.R#L55
    # CAUTION: GPL-3 license...
    df_meta = pd.read_csv("NovelViruses.csv")
    df_meta["SequenceID"] = df_meta["SequenceID"].astype("string")
    print("df_meta:\n", df_meta)
    df_meta.info()
    df = df.merge(df_meta,
                 how="right",
                 left_on="accession",
                 right_on="SequenceID",
                 ).drop(columns=["accession", "notes", "data_source", "host"])
    # drop duplicated viral species as well
    df.drop_duplicates(subset=["Name"], inplace=True)

    # from personal correspondence with Nardus (see:
    # https://re-git.lanl.gov/treddy/ldrd_virus_work/-/issues/27#note_505320)
    # it looks like I'll need to apply some filtering similar to the code
    # here (to get the same dataset they used for Figure 3):
    # https://github.com/Nardus/zoonotic_rank/blob/42f15a07ffdfc1ba425741233009f9c61bb3bf48/Scripts/Plotting/MakeFigure3.R#L62
    # CAUTION: GPL-3 license...
    df = df[(df["class"].isna()) | (df["class"] == "Mammalia") | (df["class"] == "Aves") | (df["order"] == "Diptera") |
            (df["order"] == "Ixodida")]
    df = df[df["Name"] != "Vaccinia virus"]
    assert df.duplicated(subset=["SequenceID"]).sum() == 0
    assert df.duplicated(subset=["Name"]).sum() == 0


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

    # the authors state in the manuscript that for this case study there were 758 unique
    # viral *species*

    # we're going to want the final dataframe/dataset to include
    # the primary sequence data, because that is what the ML model
    # in the main control flow uses for classification decisions
    Entrez.email = "treddy@lanl.gov"
    accessions = df["SequenceID"]
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
    df.set_index("SequenceID", inplace=True)
    for record in tqdm(records):
        df.loc[record.id, "Genome Sequence"] = record.seq
        # might as well add % GC content for now as well,
        # since that is what the current early-stage classifier
        # really uses
        df.loc[record.id, "Genome %GC"] = GC(record.seq)

    handle.close()
    print("new df:", df)
    assert df.shape[0] == 758
    print("**** filtered df that passed all assertions ***:\n", df)
    df.info()
    end = time.perf_counter()
    elapsed = end - start
    print(f"analysis complete in {elapsed:.1f} seconds")
