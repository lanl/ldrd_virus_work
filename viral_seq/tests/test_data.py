from viral_seq.data.fix_virus_names import fix_virus_names
from viral_seq.data.add_surface_exposed import add_surface_exposed
from pandas.testing import assert_frame_equal
import pandas as pd
import pytest
import numpy as np


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


@pytest.mark.parametrize(
    "surface_exposed_dict, exp_out, side_effect_in",
    [
        # this test case checks that the user input values are indexed as expected
        (
            {
                "virus_names": {
                    0: "Virus_1",
                    1: "Virus_2",
                    2: "Virus_3",
                    3: "Virus_4",
                    4: "Virus_5",
                    5: "Virus_6",
                },
                "protein_names": {
                    0: "polymerase",
                    1: "RNA",
                    2: "surface exposed protein 2",
                    3: "neuraminidase",
                    4: "surface exposed protein",
                    5: "skip protein reference",
                },
                "surface_exposed_status": {
                    0: pd.NA,
                    1: pd.NA,
                    2: pd.NA,
                    3: pd.NA,
                    4: pd.NA,
                    5: pd.NA,
                },
                "reference": {
                    0: pd.NA,
                    1: pd.NA,
                    2: pd.NA,
                    3: pd.NA,
                    4: pd.NA,
                    5: pd.NA,
                },
            },
            {
                "virus_names": {
                    0: "Virus_1",
                    1: "Virus_2",
                    2: "Virus_3",
                    3: "Virus_4",
                    4: "Virus_5",
                    5: "Virus_6",
                },
                "protein_names": {
                    0: "polymerase",
                    1: "RNA",
                    2: "surface exposed protein 2",
                    3: "neuraminidase",
                    4: "surface exposed protein",
                    5: "skip protein reference",
                },
                "surface_exposed_status": {
                    0: "no",
                    1: "no",
                    2: "yes",
                    3: "yes",
                    4: "yes",
                    5: "no",
                },
                "reference": {0: None, 1: None, 2: None, 3: None, 4: "ref_1", 5: None},
            },
            ["yes", "ref_1", "no", None],
        ),
        # this test checks that the "exit" input value works as expected
        (
            {
                "virus_names": {
                    0: "Virus_1",
                    1: "Virus_2",
                    2: "Virus_3",
                    3: "Virus_4",
                    4: "Virus_5",
                },
                "protein_names": {
                    0: "polymerase",
                    1: "RNA",
                    2: "surface exposed protein 2",
                    3: "neuraminidase",
                    4: "surface exposed protein",
                },
                "surface_exposed_status": {
                    0: pd.NA,
                    1: pd.NA,
                    2: pd.NA,
                    3: pd.NA,
                    4: pd.NA,
                },
                "reference": {0: pd.NA, 1: pd.NA, 2: pd.NA, 3: pd.NA, 4: pd.NA},
            },
            {
                "virus_names": {
                    0: "Virus_1",
                    1: "Virus_2",
                    2: "Virus_3",
                    3: "Virus_4",
                    4: "Virus_5",
                },
                "protein_names": {
                    0: "polymerase",
                    1: "RNA",
                    2: "surface exposed protein 2",
                    3: "neuraminidase",
                    4: "surface exposed protein",
                },
                "surface_exposed_status": {
                    0: "no",
                    1: "no",
                    2: "yes",
                    3: "yes",
                    4: pd.NA,
                },
                "reference": {0: None, 1: None, 2: None, 3: None, 4: np.nan},
            },
            ["exit"],
        ),
    ],
)
def test_add_surface_exposed(
    mocker, tmpdir, surface_exposed_dict, exp_out, side_effect_in
):
    exp_df = pd.DataFrame(exp_out)
    surface_exposed_df = pd.DataFrame(surface_exposed_dict)
    surface_exposed_list = ["surface exposed protein 2"]
    mocker.patch(
        "builtins.input",
        side_effect=side_effect_in,
    )

    with tmpdir.as_cwd():
        add_surface_exposed(surface_exposed_df, surface_exposed_list, "test_df.csv")
        # load saved file and check contents
        saved_df = pd.read_csv("test_df.csv")

    assert_frame_equal(saved_df, exp_df)


def test_add_surface_exposed_type_error():
    # this test checks that the function throws a TypeError for checking 'str' dtype of protein query
    surface_exposed_df = pd.DataFrame(
        {
            "virus_names": {
                0: "Virus_1",
            },
            "protein_names": {
                0: ["protein1", "protein2"],
            },
            "surface_exposed_status": {
                0: pd.NA,
            },
        }
    )
    with pytest.raises(
        TypeError, match="Invalid protein query type: expected 'str' value."
    ):
        add_surface_exposed(surface_exposed_df, [], "test_df.csv")
