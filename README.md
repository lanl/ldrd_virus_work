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
