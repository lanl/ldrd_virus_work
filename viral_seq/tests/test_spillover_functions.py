import viral_seq.analysis.spillover_predict as sp
import pandas as pd
from importlib.resources import files
import pytest

csv_train = files("viral_seq.tests").joinpath("TrainingSet.csv")
csv_test = files("viral_seq.tests").joinpath("TestSet.csv")


@pytest.mark.slow
def test_network(tmp_path):
    df_train = pd.read_csv(csv_train)
    df_test = pd.read_csv(csv_test)
    accessions_train = (" ".join(df_train["Accessions"].values)).split()
    accessions_test = (" ".join(df_test["Accessions"].values)).split()
    this_cache = tmp_path.absolute()
    # retrieve records from online and store in a cache for later use
    email = "arhall@lanl.gov"
    search_terms = accessions_test + accessions_train
    results = sp.run_search(search_terms, 1, email)
    records = sp.load_results(results, email)
    sp.add_to_cache(records, cache=this_cache)


def test_modelling():
    df_train = pd.read_csv(csv_train)
    df_test = pd.read_csv(csv_test)
    this_cache = files("viral_seq.tests") / "cache"

    # build data table of training data
    table_train = sp.build_table(
        df_train, cache=this_cache, genomic=True, gc=True, kmers=True, kmer_k=2
    )
    X_train, y_train = sp.get_training_columns(table_train)
    # cross validate
    aucs = sp.cross_validation(X_train, y_train, splits=2)
    assert aucs == pytest.approx([0.5, 0.5])

    # currently need a RandomForestClassifier to properly build a data table for the test set
    rfc = sp.train_rfc(X_train, y_train)

    # build data table of test data and run predict
    table_test = sp.build_table(
        df_test, rfc=rfc, cache=this_cache, genomic=True, gc=True, kmers=True, kmer_k=2
    )
    X_test, y_test = sp.get_training_columns(table_test)
    this_auc = sp.predict_rfc(X_test, y_test, rfc=rfc)
    assert this_auc == pytest.approx(0.3)
