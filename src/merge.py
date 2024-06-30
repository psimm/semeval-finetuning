from .common import app
from .train import merge


@app.local_entrypoint()
def merge_main(run_folder, output_dir):
    merge.remote(run_folder, output_dir)
