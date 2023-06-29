from viral_seq.analysis.get_features import get_genomic_features
from pathlib import Path
from viral_seq.analysis.spillover_predict import _append_recs
from time import perf_counter
import pandas as pd


def test_features():
    accession_path = Path("viral_seq/tests")
    test_record = _append_recs(accession_path)
    start = perf_counter()
    result = get_genomic_features(test_record)
    end = perf_counter()
    print("Calculated genomic features for one record in", end - start, "s")
    df_test = pd.read_csv("viral_seq/tests/MERS-CoV_features.csv", sep="\t")
    df = pd.DataFrame(result, index=["NC_019843.3"]).reset_index()
    # Check our features match published results for NC_019843.3 (MERS-CoV)
    for column in df_test:
        test_val = df_test[column].values[0]
        res_val = df[column].values[0]
        if isinstance(res_val, str):
            assert test_val == res_val
        elif isinstance(res_val, float):
            # Comparison data only goes to 9 decimal places
            assert test_val == float(format(res_val, ".9f"))
