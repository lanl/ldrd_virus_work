"""
The purpose of this module is to address the request of reviewer
2 described at:
https://github.com/lanl/ldrd_virus_work/issues/48
"""

import os
import numpy as np
import pandas as pd
from make_data_summary_plots import plot_family_heatmap, relative_entropy_viral_families
from tqdm import tqdm
from taxonomy_ranks import TaxonomyRanks


def assign_viral_families(df):
    supersedes = {
        "Enhydra lutris polyomavirus 1": "Polyomaviridae"  # https://www.genome.jp/virushostdb/1552409
    }
    corrections = {
        "Primate loriparvovirus 1": "Parvoviridae",
        "Pinniped copiparvovirus 1": "Parvoviridae",
        "Elephantid betaherpesvirus 5": "Orthoherpesviridae",  # https://en.wikipedia.org/wiki/Elephantid_betaherpesvirus_5
        "Elephantid betaherpesvirus 4": "Orthoherpesviridae",  # https://en.wikipedia.org/wiki/Elephantid_betaherpesvirus_4
        "Carnivore protoparvovirus": "Parvoviridae",  # https://en.wikipedia.org/wiki/Carnivore_protoparvovirus_1
        "Bat mastadenovirus H": "Adenoviridae",  # https://ictv.global/report/chapter/adenoviridae/adenoviridae
        "Bat mastadenovirus J": "Adenoviridae",  # https://ictv.global/report/chapter/adenoviridae/adenoviridae
        "Avian metaavulavirus 8": "Paramyxoviridae",  # https://ictv.global/report/chapter/paramyxoviridae/paramyxoviridae/metaavulavirus
        "Drosophina B birnavirus": "Birnaviridae",  # https://www.catalogueoflife.org/data/taxon/BXC4P
        "Goose coronavirus CB17": "Coronaviridae",  # https://www.catalogueoflife.org/data/taxon/6KPVH
        "Saint Valerien virus": "Caliciviridae",  # https://www.catalogueoflife.org/data/taxon/4TZKC
        "Salobo phlabovirus": "Phenuiviridae",  # Appears to be a typo, https://www.catalogueoflife.org/data/taxon/BXHLL
        "Tai Forest hepatitis B virus": "Hepadnaviridae",  # https://www.catalogueoflife.org/data/taxon/54K2X
        "Torque teno seal virus 1": "Anelloviridae",  # https://doi.org/10.1007/s00705-021-05192-x
        "Torque teno seal virus 2": "Anelloviridae",  # https://doi.org/10.1007/s00705-021-05192-x
        "Torque teno seal virus 3": "Anelloviridae",  # https://doi.org/10.1007/s00705-021-05192-x
        "Torque teno seal virus 8": "Anelloviridae",  # https://doi.org/10.1007/s00705-021-05192-x
        "Torque teno seal virus 9": "Anelloviridae",  # https://doi.org/10.1007/s00705-021-05192-x
    }
    unique_viral_families = [
        "Rhabdoviridae",
        "Papillomaviridae",
        "Picornaviridae",
        "Polyomaviridae",
        "Peribunyaviridae",
        "Flaviviridae",
        "Circoviridae",
        "Orthoherpesviridae",
        "Anelloviridae",
        "Adenoviridae",
        "Paramyxoviridae",
        "Genomoviridae",
        "Phenuiviridae",
        "Coronaviridae",
        "Poxviridae",
        "Arenaviridae",
        "Retroviridae",
        "Parvoviridae",
        "Sedoreoviridae",
        "Hantaviridae",
        "Spinareoviridae",
        "Togaviridae",
        "Arteriviridae",
        "Hepadnaviridae",
        "Astroviridae",
        "Nairoviridae",
        "Caliciviridae",
        "Filoviridae",
        "Orthomyxoviridae",
        "Tobaniviridae",
        "Birnaviridae",
        "Bornaviridae",
        "Hepeviridae",
        "Phasmaviridae",
        "Pneumoviridae",
        "Picobirnaviridae",
        "Asfarviridae",
        "Matonaviridae",
        "Sunviridae",
    ]
    record_family_assignments = []
    for record in df.iterrows():
        species_name = record[1].Species
        rank_taxon = TaxonomyRanks(species_name)
        if species_name in supersedes:
            viral_family = supersedes[species_name]
        else:
            try:
                rank_taxon.get_lineage_taxids_and_taxanames()
                viral_family = list(rank_taxon.lineages.values())[0]["family"][0]
            except ValueError:
                viral_family = corrections[species_name]
        if viral_family == "NA":
            viral_family = corrections[species_name]
        record_family_assignments.append(unique_viral_families.index(viral_family))
    df["viral_family"] = record_family_assignments
    return df


def randomly_shuffle_data(rng: int):
    rng = np.random.default_rng(rng)
    viral_family_indices = np.arange(39)
    rng.shuffle(viral_family_indices)
    # give a few more viral families to training than test:
    training_indices = viral_family_indices[:24]
    testing_indices = viral_family_indices[24:]
    df_training = pd.read_csv("Mollentze_Training_Fixed.csv")
    df_test = pd.read_csv("Mollentze_Holdout_Fixed.csv")
    combined = pd.concat([df_training, df_test], ignore_index=True)
    combined = assign_viral_families(df=combined)
    assert combined.viral_family.max() == 38
    assert combined.viral_family.min() == 0
    df_training_shuffled = combined[combined.viral_family.isin(training_indices)]
    df_testing_shuffled = combined[combined.viral_family.isin(testing_indices)]
    df_training_shuffled.to_csv("Mollentze_Training_Fixed_shuffled.csv", index=False)
    df_testing_shuffled.to_csv("Mollentze_Holdout_Fixed_shuffled.csv", index=False)


def check_and_retain_shuffled_datasets(relative_entropy: float, trial: int):
    "Mollentze_Training_Fixed_shuffled.csv"
    if relative_entropy >= 3.0:
        print(f"{relative_entropy=} for {trial=}; retaining CSV files")
        os.rename(
            "Mollentze_Training_Fixed_shuffled.csv",
            f"Mollentze_Training_Fixed_shuffled_{trial}.csv",
        )
        os.rename(
            "Mollentze_Holdout_Fixed_shuffled.csv",
            f"Mollentze_Holdout_Fixed_shuffled_{trial}.csv",
        )


def main(num_attempts: int = 10):
    # make num_attempts tries to reshuffle the data
    # between training and test, and preserve the subset of
    # trials that preserve a sufficiently high relative entropy
    # such that the new datasets preserve the phylogenetic distance
    # between training and test originally used by Mollentze
    for trial in tqdm(range(num_attempts)):
        # shuffle data and write candidate CSV files
        randomly_shuffle_data(rng=trial)
        # heatmap processing (we need the resulting CSV file
        # for relative entropy calculation)
        plot_family_heatmap(
            "Mollentze_Training_Fixed_shuffled.csv",
            "Mollentze_Holdout_Fixed_shuffled.csv",
        )
        relative_entropy = relative_entropy_viral_families("plot_family_heatmap.csv")
        # rename and retain the shuffled CSV files if they have sufficient
        # entropy of viral family distribution
        check_and_retain_shuffled_datasets(
            relative_entropy=relative_entropy, trial=trial
        )


if __name__ == "__main__":
    main(10)
