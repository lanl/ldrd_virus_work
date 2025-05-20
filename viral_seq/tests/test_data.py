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
    "surface_exposed_dict, exp_out, side_effect_in, check_list, explicit",
    [
        # this test case checks that the user input values are indexed as expected
        (
            {
                "surface_exposed_status": [pd.NA] * 5,
                "reference": [pd.NA] * 5,
            },
            {
                "surface_exposed_status": ["no", "no", "yes", "yes", "no"],
                "reference": [
                    "labeling performed programmatically using 'not_exposed' list",
                    "labeling performed programmatically using 'not_exposed' list",
                    "labeling performed programmatically using 'exposed' list",
                    "ref_1",
                    None,
                ],
            },
            ["yes", "ref_1", "no", None],
            None,
            False,
        ),
        # this test checks that the "exit" input value works as expected
        (
            {
                "surface_exposed_status": [pd.NA] * 5,
                "reference": [pd.NA] * 5,
            },
            {
                "surface_exposed_status": ["no", "no", "yes", pd.NA, pd.NA],
                "reference": [
                    "labeling performed programmatically using 'not_exposed' list",
                    "labeling performed programmatically using 'not_exposed' list",
                    "labeling performed programmatically using 'exposed' list",
                    np.nan,
                    np.nan,
                ],
            },
            ["exit"],
            None,
            False,
        ),
        # this test case checks that the ``check_entries`` flag works as intended
        # when passing the ``explicit=True`` flag
        (
            {
                "surface_exposed_status": ["yes", "yes", "no", "no", "no"],
                "reference": [pd.NA] * 5,
            },
            {
                "surface_exposed_status": ["no", "no", "yes", "yes", "no"],
                "reference": ["ref_1", "ref_2", "ref_3", "ref_4", None],
            },
            [
                "fix",
                "no",
                "ref_1",
                "fix",
                "no",
                "ref_2",
                "fix",
                "yes",
                "ref_3",
                "fix",
                "yes",
                "ref_4",
                "",
            ],
            [
                "polymerase",
                "RNA",
                "neuraminidase",
                "surface exposed protein",
                "skip protein reference",
            ],
            True,
        ),
        # this test case checks that the ``check_entries`` flag works as intended
        # when passing the ``explicit=False`` flag
        (
            {
                "surface_exposed_status": ["yes", "yes", "no", "no", "no"],
                "reference": [pd.NA] * 5,
            },
            {
                "surface_exposed_status": ["no", "no", "yes", "yes", "no"],
                "reference": ["ref_1", "ref_2", "ref_3", "ref_4", None],
            },
            [
                "fix",
                "no",
                "ref_1",
                "fix",
                "no",
                "ref_2",
                "fix",
                "yes",
                "ref_3",
                "fix",
                "yes",
                "ref_4",
                "",
            ],
            ["pol", "RNA", "neuraminidase", "exposed protein", "skip protein"],
            False,
        ),
    ],
)
def test_add_surface_exposed(
    mocker,
    tmpdir,
    surface_exposed_dict,
    exp_out,
    side_effect_in,
    check_list,
    explicit,
):
    virus_protein_pairs = {
        "virus_names": [f"Virus_{i}" for i in range(1, 6)],
        "protein_names": [
            "polymerase",
            "RNA",
            "neuraminidase",
            "surface exposed protein",
            "skip protein reference",
        ],
    }
    surface_exposed_dict = {**virus_protein_pairs, **surface_exposed_dict}
    exp_out = {**virus_protein_pairs, **exp_out}
    exp_df = pd.DataFrame(exp_out)
    surface_exposed_df = pd.DataFrame(surface_exposed_dict)
    mocker.patch(
        "builtins.input",
        side_effect=side_effect_in,
    )

    with tmpdir.as_cwd():
        add_surface_exposed(
            surface_exposed_df,
            "test_df.csv",
            check_entries=check_list,
            explicit=explicit,
        )
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
        add_surface_exposed(surface_exposed_df, "test_df.csv")
