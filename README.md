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

**Running the Workflow**
When runnning workflow for the first time, skip to step 2.

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
As the full workflow is not automatically tested; it should be occassionally tested locally following the above procedure, but with the `--debug` flag for `viral_seq/run_workflow.py` which will run the entire workflow with assertions on generated data which are not designed to be performative. It is pertinent to test both workflow options as they require different assertions.


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


About Licensing
===============

At the time of writing we are currently bound to the copyleft `GPL-3.0`
license because we leverage `taxonomy-ranks` in the small corner of our
workflow that deals with phylogenetic heatmaps,
and `taxonomy-ranks` itself depends on the `GPL-3.0` licensed
`ete` project. Given the minor role these libraries play in our workflow,
we'd appreciate help in finding a more liberally-licensed alternative so
that we can avoid copyleft requirements in the future.
