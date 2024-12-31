from viral_seq.data import make_alternate_datasets as make_alts
import pandas as pd
from importlib.resources import files
import pytest


# Utilizing some pre-existing accessions in the test folder
@pytest.mark.parametrize(
    "cache, accession, train_accessions, expected_result",
    [
        (
            "cache",
            "NC_033698.1",
            None,
            {"partial": [], "duplicate": [], "no_good_cds": []},
        ),
        (
            "cache",
            "HM119401.1",
            {"NC_033698.1"},
            {"partial": [], "duplicate": [], "no_good_cds": []},
        ),
        (
            "cache",
            "NC_033698.1",
            {"NC_033698.1"},
            {"partial": [], "duplicate": [0], "no_good_cds": []},
        ),
        (
            "cache_unfiltered",
            "HM147992.1",
            None,
            {"partial": [0], "duplicate": [], "no_good_cds": []},
        ),
        (
            "caches/no_good_cds",
            "KY312541.1",
            None,
            {"partial": [0], "duplicate": [], "no_good_cds": [0]},
        ),
    ],
)
def test_get_bad_indexes(cache, accession, train_accessions, expected_result):
    this_cache = files("viral_seq.tests") / cache
    df = pd.DataFrame(
        [["0", accession, "Virus"]], columns=["Unnamed: 0", "Accessions", "Species"]
    )
    result = make_alts.get_bad_indexes(df, this_cache, train_accessions)
    for key in result:
        assert result[key] == expected_result[key]
