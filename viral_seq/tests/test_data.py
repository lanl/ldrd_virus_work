from viral_seq.data.fix_virus_names import fix_virus_names
from pandas.testing import assert_frame_equal
import pandas as pd


def test_fix_virus_names():
    # make 'df' that contains column 'column_name'
    virus_data = {"Virus_Name": ["virus_0", "virus_1", "virus_2", "virus_3"]}
    df = pd.DataFrame.from_dict(virus_data)
    # make new_names that has columns 'old_name' and 'new_name'
    new_names = {
        "old_name": ["virus_0", "virus_0", "virus_1", "virus_2"],
        "new_name": ["virus_a", "virus_a", "virus_b", "virus_c"],
    }
    new_names = pd.DataFrame.from_dict(new_names)
    # expected df
    out_expected = {"Virus_Name": ["virus_a", "virus_b", "virus_c", "virus_3"]}
    out_df_expected = pd.DataFrame.from_dict(out_expected)
    # function call
    out_df = fix_virus_names(df, new_names, column_name="Virus_Name")
    assert_frame_equal(out_df, out_df_expected)
