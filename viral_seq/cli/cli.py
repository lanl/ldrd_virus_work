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
@click.option('--save_model',
              is_flag=True,
              show_default=True,
              default=False,
              help=("Serialize (pickle) the trained ML model(s) to"
                    " disk so that they can be packaged/reused elsewhere"))
@click.option('--load_model',
              is_flag=True,
              show_default=True,
              default=False,
              help=("Load the trained (serialized/pickled) ML model(s) in to"
                    " memory so that they can be used immediately in cases where"
                    " it is not necessary to iterate on the model proper. This will"
                    " require that `save_model` has been used locally first."))
def viral_seq(download_recs,
              save_model,
              load_model):
    start_sec = perf_counter()
    main(download_recs=download_recs,
         save_model=save_model,
         load_model=load_model)
    end_sec = perf_counter()
    execution_time_sec = end_sec - start_sec
    print(f"viral_seq execution time (s): {execution_time_sec:.2f}")
