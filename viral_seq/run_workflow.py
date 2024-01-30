from importlib.resources import files
from itertools import filterfalse
from viral_seq.analysis import spillover_predict as sp
from viral_seq.cli import cli
import pandas as pd
import re
import argparse
from tqdm import tqdm


def find_in_record(this_search, record_folder):
    genbank_file = list(record_folder.glob("*.genbank"))[0]
    with open(genbank_file, "r") as f:
        line = next((s for s in f if bool(re.search(this_search, s))), None)
    if line is None:
        return False
    else:
        return True


def build_cache(generate=True, debug=False):
    """Download and store all data needed for the workflow"""

    if generate:
        print(
            "Will pull down test and train data to local cache, as well as human gene data for similarity features."
        )

    if debug:
        print("Debug mode: will run assertions on generated cache")

    email = "arhall@lanl.gov"
    data = files("viral_seq.data")
    cache_Trav = data / "cache_viral"
    train_Trav = data.joinpath("Mollentze_Training.csv")
    test_Trav = data.joinpath("Mollentze_Holdout.csv")
    cache_viral = cache_Trav.absolute().as_posix()
    viral_files = [train_Trav.absolute().as_posix(), test_Trav.absolute().as_posix()]

    if generate:
        print("Pulling viral sequence data to local cache...")
        for file in viral_files:
            cli.pull_data(
                [
                    "--email",
                    email,
                    "--cache",
                    cache_viral,
                    "--file",
                    file,
                    "--no-filter",
                ],
                standalone_mode=False,
            )

    if debug:
        print("Validating train and test cache...")
        df = pd.read_csv(viral_files[0])
        accessions_train = set([s for s in (" ".join(df["Accessions"].values)).split()])
        df = pd.read_csv(viral_files[1])
        accessions_test = set([s for s in (" ".join(df["Accessions"].values)).split()])
        cache_path, cache_subdirectories = sp.init_cache(cache_viral)
        cached_accessions = set(s.parts[-1] for s in cache_subdirectories)
        cached_accessions_stems = set(s.split(".")[0] for s in cached_accessions)
        cached_train = cached_accessions.intersection(accessions_train)
        cached_test = cached_accessions.intersection(accessions_test)
        # assert what's missing just differs by version with something present
        for this_set, that_set in zip(
            [cached_train, cached_test], [accessions_train, accessions_test]
        ):
            missing = that_set.difference(this_set)
            if len(missing) > 0:
                for each in missing:
                    this_stem = each.split(".")[0]
                    assert this_stem in cached_accessions_stems
                    print(
                        each,
                        "not in cache, but suitable accession present:",
                        [s for s in cached_accessions if s.startswith(this_stem)][0],
                    )
        # assert we don't have anything extra
        assert len(accessions_train.union(accessions_test)) == len(cache_subdirectories)

    cache_Trav = data / "cache_housekeeping"
    housekeeping_Trav = data.joinpath("Housekeeping_accessions.txt")
    cache_hk = cache_Trav.absolute().as_posix()
    file = housekeeping_Trav.absolute().as_posix()

    if generate:
        print(
            "Pulling sequence data of human housekeeping genes for similarity features..."
        )
        with open(file, "r") as f:
            transcripts = f.readlines()[0].split()
        search_term = "(biomol_mrna[PROP] AND refseq[filter]) AND("
        first = True
        for transcript in transcripts:
            if first:
                first = False
            else:
                search_term += "OR "
            search_term += transcript + "[Accession] "
        search_term += ")"
        results = sp.run_search(
            search_terms=[search_term], retmax=len(transcripts), email=email
        )
        records = sp.load_results(results, email=email)
        sp.add_to_cache(records, just_warn=True, cache=cache_hk)
    if debug:
        print("Validating human housekeeping gene cache...")
        with open(file, "r") as f:
            accessions_hk = set([s.split(".")[0] for s in (f.readlines()[0]).split()])
        cache_path, cache_subdirectories = sp.init_cache(cache_hk)
        cached_accessions = set(s.parts[-1].split(".")[0] for s in cache_subdirectories)
        missing_hk_file = data / "missing_hk.txt"
        cached_hk = cached_accessions.intersection(accessions_hk)
        missing_hk = accessions_hk.difference(cached_hk)
        extra_accessions = cached_accessions.difference(accessions_hk)
        if len(missing_hk) > 0:
            renamed_accessions = {}
            # some accessions have been replaced
            print("Checking if missing accessions have been renamed...")
            for accession in missing_hk:
                search_term = (
                    "(biomol_mrna[PROP] AND refseq[filter]) AND "
                    + accession
                    + "[Accession]"
                )
                results = sp.run_search(
                    search_terms=[search_term], retmax=1, email=email
                )
                if len(results) > 0:
                    assert len(results) == 1
                    new_name = list(results)[0].split(".")[0]
                    print(accession, "was renamed as", new_name)
                    renamed_accessions[accession] = new_name
            for k, v in renamed_accessions.items():
                missing_hk.remove(k)
                extra_accessions.remove(v)
            assert len(extra_accessions) == 0
            # currently we don't expect to find everything
            print(
                "Couldn't find",
                len(missing_hk),
                "accessions for housekeeping genes; saved to",
                missing_hk_file.absolute().as_posix(),
            )
            with open(missing_hk_file, "w") as f:
                print("Missing accessions for human housekeeping genes", file=f)
                print(missing_hk, file=f)
        # TODO: stricter assertion when we expect to find all accessions
        # assert we don't have anything extra
        assert len(cached_accessions) + len(missing_hk) == len(accessions_hk)

    cache_Trav = data / "cache_isg"
    isg_Trav = data.joinpath("ISG_transcript_ids.txt")
    cache_isg = cache_Trav.absolute().as_posix()
    file = isg_Trav.absolute().as_posix()
    if generate:
        print("Pulling sequence data of human isg genes for similarity features...")
        cli.pull_ensembl_transcripts(
            [
                "--email",
                email,
                "--cache",
                cache_isg,
                "--file",
                file,
            ],
            standalone_mode=False,
        )

    if debug:
        print("Validating human ISG gene cache...")
        with open(file, "r") as f:
            ensembl_ids = set((f.readlines()[0]).split())
        cache_path, cache_subdirectories = sp.init_cache(cache_isg)
        total_cached = len(cache_subdirectories)
        missing = []
        missing_file = data / "missing_isg.txt"
        # see if we can find the ensembl transcript ids in the raw text of the genbank files (this is slow)
        for this_id in tqdm(ensembl_ids):
            # make our work easier as we go
            new_list = list(
                filterfalse(
                    lambda e: find_in_record(r"MANE Ensembl match\s+:: " + this_id, e),
                    cache_subdirectories,
                )
            )
            if len(new_list) == len(cache_subdirectories):
                missing.append(this_id)
            else:
                # if we had a match we should have one less directory to look at
                assert len(new_list) == len(cache_subdirectories) - 1
            cache_subdirectories = new_list[:]
        # TODO: make this more strict if we expect to find all the transcripts
        # assert we don't have anything extra
        assert len(cache_subdirectories) == 0
        assert len(missing) + total_cached == len(ensembl_ids)
        print(
            "Couldn't find",
            len(missing),
            "transcripts; ids saved to",
            missing_file.absolute().as_posix(),
        )
        with open(missing_file, "w") as f:
            print("Missing ISG transcripts", file=f)
            print(missing, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cache", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    cache = args.cache
    debug = args.debug

    build_cache(generate=cache, debug=debug)
