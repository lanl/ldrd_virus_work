import viral_seq.analysis.spillover_predict as sp


def search_handling():
    email = "arhall@lanl.gov"
    # Only two search terms so we don't tax the search just for testing
    search_terms = ["NC_045512.2", "NC_019843.3"]
    results = sp.run_search(search_terms, 1, email)
    # as we looked for the accessions, the results should exactly match
    print("Accessions retrieved:", results)
    for result in results:
        assert result in search_terms
    # load into memory
    records = sp.load_results(results, email)
    # check they are the right accessions again
    record_ids = []
    for record in records:
        print("record.id of loaded", record.id)
        assert record.id in search_terms
        record_ids.append(record.id)
    # run record filters
    records = sp.filter_records(records)
    # also just assert them
    sp.filter_records(records, just_assert=True)
    # try to add to cache but this should fail as they already exist
    sp.add_to_cache(records)
    # lets just load the records from the cache now, and check they match the online record
    records_cached = sp.load_from_cache(results)
    for record in records_cached:
        print("Checking", record.id, "in cache matches record pulled from online")
        assert record.id in record_ids


def test_modelling():
    print("Loading entire cache to test model building")
    records_cached = sp.load_from_cache()
    df = sp.build_table(records_cached)
    X, y = sp.get_training_columns(df)
    sp.cross_validation_random_forest(X, y, plot=True)
    clf = sp.train_random_forest(X, y)
