from viral_seq.analysis import dtra_utils
import pandas as pd
from numpy.testing import assert_array_equal


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
