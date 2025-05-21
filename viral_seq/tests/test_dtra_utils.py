from viral_seq.analysis import dtra_utils
import pandas as pd
from numpy.testing import assert_array_equal
from importlib.resources import files


def test_get_surface_exposure_status():
    # make lists of virus and protein names
    viruses = ["virus_1", "virus_2", "virus_2", "virus_3"]
    proteins = ["protein_1", "protein_1", "protein_2", "protein_2"]
    # make a dataframe of surface exposure status containing the columns ``virus_names`` and ``protein_names`` and ``surface_exposed_status``
    surface_exposed_df = pd.DataFrame(
        {
            "virus_names": [
                "virus_1",
                "virus_1",
                "virus_2",
                "virus_2",
                "virus_3",
                "virus_3",
            ],
            "protein_names": ["protein_1", "protein_2"] * 3,
            "surface_exposed_status": ["yes", "no"] * 3,
        }
    )

    exp_out = ["yes", "yes", "no", "no"]

    surface_exposed_list = dtra_utils.get_surface_exposure_status(
        viruses, proteins, surface_exposed_df
    )

    assert_array_equal(surface_exposed_list, exp_out)


def test_merge_convert_tbl():
    igsf_data = files("viral_seq.data") / "igsf_training.csv"
    input_csv = files("viral_seq.data") / "receptor_training.csv"

    merged_df = dtra_utils.merge_tables(input_csv, igsf_data)
    converted_df = dtra_utils.convert_merged_tbl(merged_df)

    assert merged_df.shape == (118, 8)
    assert converted_df.shape == (118, 10)
    assert converted_df.IN.sum() == 45
    assert converted_df.SA.sum() == 53
    assert converted_df.SA_IG.sum() == 6
    assert converted_df.IN_IG.sum() == 7
    assert converted_df.IN_SA.sum() == 2
    assert converted_df.IN_SA_IG.sum() == 2
    assert converted_df.IG.sum() == 35
