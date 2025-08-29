# ldrd_virus_work (LANL copyright assertion ID: O# (O4909))

## LDRD DR Computational Work Repo

**Pre-Commit Procedure**
Linting pre-commit procedure prevents unnecessary CI/CD failures, but testing procedure is necessary as tests marked slow will not run in CI/CD. These must be run in pre-commit.

- Linting
```
>black viral_seq
>ruff check viral_seq --fix
>mypy -p viral_seq
```

- Testing
```
>cd /tmp
>python3 -m pytest --pyargs viral_seq
```

Workflow Memory Requirements
============================

Running the full LDRD workflow can be quite memory intensive--we have found that it
can often be necessary to increase the amount of swap available on a Linux system
to avoid memory-related failures during the SHAP-based determination of features
to be retained in the workflow. Increasing swap from 2 GB to 40 GB was recently
effective on one problematic Linux run, even for a machine with 128 GB RAM.

**Running the Workflow**
When running workflow for the first time, skip to step 2.

1. Uninstall viral_seq with `python3 -m pip uninstall viral_seq`
2. Ensure large files have been pulled down with `git lfs pull` from root directory (for `git-lfs` installation instructions see https://git-lfs.com/)
3. Install viral_seq with `python3 -m pip install .` from root directory
4. It is advised to create and run the workflow from a fresh working directory to keep artifacts from different runs isolated
5. Run the workflow with the following commands, replacing [relative_path] as appropriate for your working directory:

Using stored cache:

```
>python3 [relative_path]/viral_seq/run_workflow.py
```

Pulling down the cache at runtime:

```
>python3 [relative_path]/viral_seq/run_workflow.py --cache 3
```

**Workflow Testing**
As the full workflow is not automatically tested; it should be occasionally tested locally following the above procedure, but with the `--debug` flag for `viral_seq/run_workflow.py` which will run the entire workflow with assertions on generated data which are not designed to be performative. It is pertinent to test both workflow options as they require different assertions.


Generating Heatmaps and Calculating Relative Entropy Quickly
============================================================

At the time of writing it can take longer than an hour to run
the full workflow that supports the paper. However, generation
of the phylogenetic heatmaps of viral family representation
in training and test datasets, and the corresponding relative
entropy calculation for those distributions, can be done quickly
with an incantation like the one below. This will error out, but
will produce the heatmap and a printout of the relative entropy
before it does:

```
> python ../viral_seq/run_workflow.py --cache 0 --features 0 --feature-selection skip -tr Mollentze_Training_Shuffled.csv -ts Mollentze_Holdout_Shuffled.csv
```

(and similarly for other training and test datasets)

Producing the Violin Plot for LDRD manuscript (and related data)
================================================================

The ROC AUC violin plot in the LDRD manuscript, which compares
relative ML model performances for human, primate, and mammal targets
(and with vs. without shuffling/rebalancing the data), can be regenerated
by running the six pertinent LDRD workflows and then running the post-processing
code.

For example, after installing the project locally and confirming that the
testsuite is passing, six subfolders for the different conditions might be
created, and the workflow incantations initiated in those directories as follows
(10 random seeds were combined for the manuscript results):

```
# (1) at subdirectory human_fixed:
python ../viral_seq/run_workflow.py -tr Relabeled_Train.csv -ts Relabeled_Test.csv -tc "human" -c "extract" -n 2 -cp 10
# (2) at subdirectory human_shuffled:
python ../viral_seq/run_workflow.py -tr Relabeled_Train_Human_Shuffled.csv -ts Relabeled_Test_Human_Shuffled.csv -tc "human" -c "extract" -n 2 -cp 10
# (3) at subdirectory primate_fixed:
python ../viral_seq/run_workflow.py -tr Relabeled_Train.csv -ts Relabeled_Test.csv -tc "primate" -c "extract" -n 2 -cp 10
# (4) at subdirectory primate_shuffled:
python ../viral_seq/run_workflow.py -tr Relabeled_Train_Primate_Shuffled.csv -ts Relabeled_Test_Primate_Shuffled.csv -tc "primate" -c "extract" -n 2 -cp 10
# (5) at subdirectory mammal_fixed:
python ../viral_seq/run_workflow.py -tr Relabeled_Train.csv -ts Relabeled_Test.csv -tc "mammal" -c "extract" -n 2 -cp 10
# (6) at subdirectory mammal_shuffled:
python ../viral_seq/run_workflow.py -tr Relabeled_Train_Mammal_Shuffled.csv -ts Relabeled_Test_Mammal_Shuffled.csv -tc "mammal" -c "extract" -n 2 -cp 10
```

After verifying that each workflow has completed normally (i.e., exit code of `0`,
no error message), the post-processing code can be started to generate the violin
plot and its associated raw data:

```python
from viral_seq.data.make_target_comparison_plot import plot_target_comparison

plot_target_comparison(
    human_fixed_predictions="human_fixed/data_calculated/predictions",
    human_shuffled_predictions="human_shuffled/data_calculated/predictions",
    primate_fixed_predictions="primate_fixed/data_calculated/predictions",
    primate_shuffled_predictions="primate_shuffled/data_calculated/predictions",
    mammal_fixed_predictions="mammal_fixed/data_calculated/predictions",
    mammal_shuffled_predictions="mammal_shuffled/data_calculated/predictions",
)   
```

Generating estimator/target statistics for LDRD manuscript
==========================================================

Average ROC AUC values and their standard deviations are typically
reported over ten different random seeds for a given condition. As
an example of how these might be generated/reproduced, consider
the condition of "human target" + "shuffled data" + "hyperparameter
optimized." In this case, we run the workflow with an incantation like
this:

`python ../viral_seq/run_workflow.py -tr Relabeled_Train_Human_Shuffled.csv -ts Relabeled_Test_Human_Shuffled.csv -tc "human" -c "extract" -n 16 -cp 10`

After the workflow completes, we aggregate the statistics with
a script that looks like this:

```python
import glob


from viral_seq.analysis import spillover_predict as sp


for pred_file in glob.glob("human_shuffled/data_calculated/predictions/*.csv"):
    print(f"{pred_file=}")
    sp.get_aucs(
                f"{pred_file}",
                "Relabeled_Test_Human_Shuffled.csv",
                # the above CSV file may change depending on the condition; some
                # other typical values include: `Relabeled_Test.csv` (non-shuffled),
                # `Relabeled_Test_Primate_Shuffled.csv` (primate shuffled),
                # `Relabeled_Test_Mammal_Shuffled.csv` (mammal shuffled)
                "human",
                # the above target host may change to a value of `mammal` or
                # `primate` or `Human Host` (for the original Mollentze data)
                )
    print("-" * 20)
```

That will print out the average and standard deviation ROC AUC of each estimator
type across the ten random seeds.

Aggregate ROC AUC statistics for LDRD manuscript
================================================

In addition to obtaining the ROC AUC statistics for each individual estimator
across several random seeds (see above), it is also possible to calculate the
overall ROC AUC statistics across all of the estimators and seeds combined, and to
compare those overall ROC AUC statistics between different workflow run conditions.
Here is an example script that covers this type of calculation/comparison,
where the two conditions being contrasted involve the human host target
with vs. without hyperparameter optimization:

```python
from viral_seq.analysis import spillover_predict as sp

sp.compare_workflow_aucs(
                    ("human_shuffled_no_opt/data_calculated/predictions",
                     "human_shuffled/data_calculated/predictions"),
                    ("Relabeled_Test_Human_Shuffled.csv",
                     "Relabeled_Test_Human_Shuffled.csv"),
                    ("human", "human"),
                    )
```

This will produce overall ROC AUC stats, like those summarized in the
abbreviated output below:

```
human_shuffled_no_opt/data_calculated/predictions:
<snip>
Workflow mean auc = 0.769 std = 0.014
========================
human_shuffled/data_calculated/predictions:
<snip>
Workflow mean auc = 0.784 std = 0.013

Student t-test: t_stat -7.022, p-value 0.000

```

Comparing Dataset vs. Mollentze for LDRD manuscript
===================================================

To compare the "human host" infection status changes
for records in our new dataset vs. the original
dataset used by Mollentze, there is a small script
at `viral_seq/data` that can be executed:

`python summarize_changes_v_mollentze.py`

Which will output a summary of the human host infection
status changes:

```
Number of records for which human infection status switched from False to True: 21
Number of records for which human infection status switched from True to False: 0
```



About Licensing
===============

At the time of writing we are currently bound to the copyleft `GPL-3.0`
license because we leverage `taxonomy-ranks` in the small corner of our
workflow that deals with phylogenetic heatmaps,
and `taxonomy-ranks` itself depends on the `GPL-3.0` licensed
`ete` project. Given the minor role these libraries play in our workflow,
we'd appreciate help in finding a more liberally-licensed alternative so
that we can avoid copyleft requirements in the future.
