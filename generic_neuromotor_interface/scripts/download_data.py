# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import click
from datetime import datetime

from generic_neuromotor_interface.download_utils import (
    download_file,
    ensure_dir,
    extract_tar,
)


def download_data(task_name, dataset, emg_data_dir):
    """
    Download data for a specific task and dataset.

    Args:
        task_name (str): Name of the task (handwriting, discrete_gestures, or wrist)
        dataset (str): Dataset type (full_data or small_subset)
        emg_data_dir (str): Directory where the data will be stored

    Returns:
        Path: Path to the directory where data was downloaded
    """
    base_url = "https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data"

    print(f"Downloading the {dataset} data for {task_name}...")

    # Create directory
    data_dir = ensure_dir(Path(emg_data_dir))

    # Check flag file (has data already been downloaded?)
    flag_file = Path(data_dir / f".data_downloaded_{dataset}_{task_name}")
    if flag_file.exists():
        modification_timestamp = flag_file.stat().st_mtime
        last_modified_datetime = datetime.fromtimestamp(modification_timestamp)
        print(
            f"Found data downloaded for {task_name=}, {dataset=} at {data_dir=}. "
            f"Last modified: {last_modified_datetime}. "
            "Assuming data already downloaded!"
        )
        return data_dir

    # Download the tar file
    tar_filename = f"{task_name}_{dataset}.tar"
    tar_path = data_dir / tar_filename
    data_url = f"{base_url}/data/{task_name}/{dataset}.tar"
    download_file(data_url, tar_path, f"Downloading {dataset} data for {task_name}")

    # Extract the tar file
    extract_tar(tar_path, data_dir, "Unzipping the data")

    # Download the corpus spreadsheet
    print("Downloading the corpus spreadsheet...")
    corpus_filename = f"{task_name}_corpus.csv"
    corpus_path = data_dir / corpus_filename
    corpus_url = f"{base_url}/data/{task_name}/{corpus_filename}"
    download_file(
        corpus_url, corpus_path, f"Downloading {task_name} corpus spreadsheet"
    )

    # Touch the flag file to signal data was downloaded
    flag_file.touch()

    print(f"Data for {task_name} ({dataset}) downloaded and extracted to {data_dir}")
    return data_dir


@click.command()
@click.argument(
    "task_name", type=click.Choice(["handwriting", "discrete_gestures", "wrist"])
)
@click.argument("dataset", type=click.Choice(["full_data", "small_subset"]))
@click.argument("emg_data_dir", type=click.Path())
def main(task_name, dataset, emg_data_dir):
    """
    Download data for neuromotor interface tasks.

    TASK_NAME: Name of the task (handwriting, discrete_gestures, or wrist)

    DATASET: Dataset type (full_data or small_subset)

    EMG_DATA_DIR: Directory where the data will be stored
    """
    download_data(task_name, dataset, emg_data_dir)


if __name__ == "__main__":
    main()
