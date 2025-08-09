"""
This module contains code for generating the violin
plot used in the LDRD manuscript. For details on
usage, see the repo README file.
"""

import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from viral_seq.analysis import spillover_predict as sp


def add_data(group, method, values, data):
    for v in values:
        data.append({"Target": group, "Dataset": method, "ROC AUC": v})


def plot_target_comparison(
    human_fixed_predictions: str,
    human_shuffled_predictions: str,
    primate_fixed_predictions: str,
    primate_shuffled_predictions: str,
    mammal_fixed_predictions: str,
    mammal_shuffled_predictions: str,
):
    data = []
    groups = ["human", "primate", "mammal"]
    methods = ["fixed", "shuffled"]
    folders = [
        human_fixed_predictions,
        human_shuffled_predictions,
        primate_fixed_predictions,
        primate_shuffled_predictions,
        mammal_fixed_predictions,
        mammal_shuffled_predictions,
    ]
    dataset_files = {
        "human": "Relabeled_Test_Human_Shuffled.csv",
        "primate": "Relabeled_Test_Primate_Shuffled.csv",
        "mammal": "Relabeled_Test_Mammal_Shuffled.csv",
    }
    for (target, method), folder in zip(product(groups, methods), folders):
        wf_files = glob.glob(os.path.join(folder, "*csv"))
        aucs = []
        for file in wf_files:
            this_dataset_file = (
                "Relabeled_Test.csv" if method == "fixed" else dataset_files[target]
            )
            print(f"{target}, {method}:")
            these_aucs = sp.get_aucs(file, this_dataset_file, target)
            aucs += these_aucs
        label = "Corrected" if method == "fixed" else "Rebalanced"
        add_data(target.capitalize(), label, aucs, data)

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.violinplot(
         data=df,
         x="Target",
         y="ROC AUC",
         hue="Dataset",
         split=True,
         gap=0.1,
         inner="quart",
         ax=ax,
    )
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.legend(title=ax.get_legend().get_title().get_text(),
              title_fontsize=14,
              alignment='center')
    for text in ax.get_legend().get_texts():
        text.set_fontsize(14)
    fig.tight_layout()
    fig.savefig("retarget_comparison.png", dpi=300)
