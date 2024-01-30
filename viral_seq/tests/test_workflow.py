from importlib.resources import files
from viral_seq import run_workflow as workflow
import pytest


@pytest.mark.parametrize(
    "search_term, result", [(r"Alenquer\svirus", True), ("spam", False)]
)
def test_find_in_record(search_term, result):
    this_record = files("viral_seq.tests") / "cache" / "HM119401.1"
    assert workflow.find_in_record(search_term, this_record) == result
