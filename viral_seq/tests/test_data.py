from viral_seq.data.fix_virus_names import fix_virus_names
from pandas.testing import assert_frame_equal
import pandas as pd
import pytest


@pytest.mark.slow
# TODO: implement mocker for `load_results` inside function call
def test_fix_virus_names():
    virus_data = {
        "Virus_Name": {0: "ross river virus", 1: "hMPV"},
        "Accessions": {0: "NC_075016.1", 1: "NC_039199.1"},
    }
    df = pd.DataFrame.from_dict(virus_data)

    out_df_expected = {
        "Virus_Name": {0: "Ross River virus", 1: "human metapneumovirus"},
        "Accessions": {0: "NC_075016.1", 1: "NC_039199.1"},
    }

    out_df = fix_virus_names(df, column_name="Virus_Name")

    assert_frame_equal(out_df, pd.DataFrame.from_dict(out_df_expected))
