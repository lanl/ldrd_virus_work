import numpy as np
import scipy
import pandas as pd
from taxonomy_ranks import TaxonomyRanks
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

matplotlib.use("Agg")


def plot_family_heatmap(
    train_file: str,
    test_file: str,
    target_column: str = "Human Host",
    filename_plot: str = "plot_family_heatmap.png",
    filename_data: str = "plot_family_heatmap.csv",
):
    """Generate a heatmap of viral family counts.

    Viral family counts are shown separated by train, test, and value of `target_column`.
    This plot allows viral family representation to be evaluated at a glance.

    Parameters:
    -----------
    train_file: str
        filepath of the train csv
    test_file: str
        filepath of the test csv
    target_column: str
        training column from dataset
    filename_plot: str
        filepath to output generated plot
    filename_data: str
        filepath to output generated plot's source data
    """
    # get family counts
    df_train = pd.read_csv(train_file, index_col=False)[["Species", target_column]]
    train_true = _get_family_counts(df_train.loc[df_train[target_column]])
    train_false = _get_family_counts(df_train.loc[~df_train[target_column]])
    df_test = pd.read_csv(test_file, index_col=False)[["Species", target_column]]
    test_true = _get_family_counts(df_test.loc[df_test[target_column]])
    test_false = _get_family_counts(df_test.loc[~df_test[target_column]])
    # format DataFrame
    family_counts = pd.DataFrame(
        [train_true, train_false, test_true, test_false],
        index=[
            f"Train {target_column}",
            f"Train Not {target_column}",
            f"Test {target_column}",
            f"Test Not {target_column}",
        ],
    ).T
    family_counts.fillna(0, inplace=True)
    family_counts = family_counts.astype("int32")
    # sort by family total across all groupings
    family_counts["sum"] = family_counts.sum(axis=1)
    family_counts["family"] = family_counts.index
    family_counts.sort_values(
        by=["sum", "family"], ascending=[False, True], inplace=True
    )
    family_counts.drop(columns=["sum", "family"], inplace=True)
    _plot_family_heatmap(family_counts.T, filename_plot, filename_data)


def _get_family_counts(df: pd.DataFrame) -> dict[str, int]:
    # These couldn't be found automatically and were looked up
    corrections = {
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

    misses = []
    families: dict[str, int] = defaultdict(int)
    for species in df["Species"].values:
        family = ""
        # taxonomy_ranks will try to look up the first word if the whole Species name doesn't return anything
        # however, it is often better to try other words in the species name
        for this_search in [species] + species.split():
            rank_taxon = TaxonomyRanks(this_search)
            try:
                rank_taxon.get_lineage_taxids_and_taxanames()
            except ValueError:
                # if nothing is found, taxonomy_ranks throws an error, but we will keep looking
                continue
            # Documentation states multiple lineages could possibly be returned https://github.com/linzhi2013/taxonomy_ranks/tree/master?tab=readme-ov-file#32-using-as-a-module
            # However, I cannot find an example of this
            if len(rank_taxon.lineages) != 1:
                raise ValueError(
                    f'Multiple lineages were returned for {species}. Please verify the name "{species}" is correct.'
                )
            key = next(iter(rank_taxon.lineages))
            family = rank_taxon.lineages[key]["family"][0]
            if "viridae" in family.lower():
                break
        # ensure we found a viral family
        if "viridae" not in family.lower():
            if species in corrections:
                # we manually looked this up already
                family = corrections[species]
            else:
                # nothing found, most likely a typo or name synonyms/discrepancies
                misses.append(species)
                family = "NOT FOUND"
        families[family] += 1

    # manually look up whatever we couldn't find
    if len(misses) > 0:
        raise ValueError(
            "Couldn't find taxonomy for the following viruses:",
            misses,
            "Manually look up the family for these viruses and put in the `corrections` dictionary",
        )
    else:
        return families


def _plot_family_heatmap(
    family_counts: pd.DataFrame,
    filename_plot: str = "plot_family_heatmap.png",
    filename_data: str = "plot_family_heatmap.csv",
):
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.get_cmap("bwr")
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    cm_wr = LinearSegmentedColormap.from_list("Upper Half", colors)
    norm = LogNorm(vmin=1, vmax=200, clip=True)
    sns.heatmap(
        family_counts,
        annot=True,
        norm=norm,
        cmap=cm_wr,
        yticklabels=True,
        square=True,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(filename_plot, dpi=300)
    plt.close()
    family_counts.to_csv(filename_data)


def relative_entropy_viral_families(heatmap_csv: str) -> float:
    """
    Calculate the relative entropy (Kullback-Leibler divergence)
    of the distribution of viral families between training and test
    datasets.

    Parameters:
    -----------
    heatmap_csv: str
        filepath for the CSV file containing viral family distribution
        data between train and test sets
    Returns:
    -----------
    relative_entropy: float
        The relative entropy (Kullback-Leibler divergence) of the distribution
        of viral families between training and test data sets.
    """
    df = pd.read_csv(heatmap_csv)
    # for the purposes of the relative entropy calculations,
    # we sum the target infecting and non-infecting viruses
    # to get the total viruses from each family in either train
    # or test splits
    train_sums = df.iloc[[0, 1]].sum().values[1:].astype(int)
    test_sums = df.iloc[[2, 3]].sum().values[1:].astype(int)
    kl = scipy.stats.entropy(pk=train_sums, qk=test_sums)
    return kl
