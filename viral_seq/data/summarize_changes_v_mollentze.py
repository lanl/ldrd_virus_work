"""
The purpose of this module is to compare
the human vs. not-human infection status of
records in our improved dataset vs. the original
Mollentze dataset. We just want a few basic measures
of what has changed vs. the original dataset.
"""

import pandas as pd


def main():
    orig_train_df = pd.read_csv(
        "Mollentze_Training.csv", usecols=["Accessions", "Human Host"]
    )
    new_train_df = pd.read_csv("Relabeled_Train.csv", usecols=["Accessions", "human"])
    false_to_true = 0
    true_to_false = 0
    for new_row in new_train_df.itertuples():
        new_accession = new_row[1]
        new_human_infection = new_row[2]
        old_human_infection = orig_train_df[
            orig_train_df["Accessions"] == new_accession
        ]["Human Host"].values.item()
        if (not old_human_infection) and new_human_infection:
            false_to_true += 1
        elif old_human_infection and (not new_human_infection):
            true_to_false += 1
    print(
        "Number of records for which human infection status switched from "
        f"False to True: {false_to_true}"
    )
    print(
        "Number of records for which human infection status switched from "
        f"True to False: {true_to_false}"
    )


if __name__ == "__main__":
    main()
