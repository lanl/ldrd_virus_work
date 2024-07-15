# ldrd_virus_work

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


