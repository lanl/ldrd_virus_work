from time import perf_counter

import rich_click as click
from viral_seq.analysis.spillover_predict import main


click.rich_click.OPTION_GROUPS = {
    "viral_seq": [
        {
            "name": "Basic usage",
            "options": ["--download_recs", "--help"],
        },
    ],
}


@click.command()
@click.option('--download_recs',
              type=int,
              show_default=True,
              default=0,
              help=("Specify a number of sequence records "
                    "to download (per search category), either generating "
                    "or expanding the local cache with records it doesn't "
                    "already have."))
def viral_seq(download_recs):
    start_sec = perf_counter()
    main(download_recs=download_recs)
    end_sec = perf_counter()
    execution_time_sec = end_sec - start_sec
    print(f"viral_seq execution time (s): {execution_time_sec:.2f}")
