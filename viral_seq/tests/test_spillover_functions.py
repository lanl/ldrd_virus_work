import viral_seq.analysis.spillover_predict as sp
import pandas as pd
from importlib.resources import files


def test_modelling(tmp_path):
    df_train = pd.read_csv(files("viral_seq.tests").joinpath("TrainingSet.csv"))
    df_test = pd.read_csv(files("viral_seq.tests").joinpath("TestSet.csv"))
    accessions_train = (" ".join(df_train["Accessions"].values)).split()
    print("training list")
    print(accessions_train)
    accessions_test = (" ".join(df_test["Accessions"].values)).split()
    print("test list")
    print(accessions_test)
    this_cache = tmp_path.absolute()
    print("temp cache location", this_cache)
    # retrieve records from online and store in a cache for later use
    email = "arhall@lanl.gov"
    search_terms = accessions_test + accessions_train
    results = sp.run_search(search_terms, 1, email)
    records = sp.load_results(results, email)
    sp.add_to_cache(records, cache=this_cache)

    # build data table of training data
    table_train = sp.build_table(
        df_train, cache=this_cache, genomic=True, gc=True, kmers=True, kmer_k=2
    )
    print("training table")
    print(table_train)
    X_train, y_train = sp.get_training_columns(table_train)
    # cross validate
    aucs = sp.cross_validation(X_train, y_train, splits=2)
    print("cross validation AUC scores:", aucs)
    # assert AUCs are as expected
    assert aucs == [0.3333333333333333, 0.5]

    # currently need a RandomForestClassifier to properly build a data table for the test set
    rfc = sp.train_rfc(X_train, y_train)

    # build data table of test data and run predict
    table_test = sp.build_table(
        df_test, rfc=rfc, cache=this_cache, genomic=True, gc=True, kmers=True, kmer_k=2
    )
    print("testing table")
    print(table_test)
    X_test, y_test = sp.get_training_columns(table_test)
    this_auc = sp.predict_rfc(X_test, y_test, rfc=rfc)
    print("test set AUC score", this_auc)
    # assert AUC is as expected
    assert this_auc == 0.38
