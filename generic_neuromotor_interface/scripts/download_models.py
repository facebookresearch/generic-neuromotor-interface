# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import click

from generic_neuromotor_interface.download_utils import (
    download_file,
    ensure_dir,
    extract_tar,
)


def download_models(task_name, model_dir):
    """
    Download pretrained models for a specific task.

    Args:
        task_name (str): Name of the task (handwriting, discrete_gestures, or wrist)
        model_dir (str): Directory where the model will be stored
    """
    base_url = "https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data"

    print(f"Downloading the pretrained model for {task_name}...")

    # Create directories
    task_dir = ensure_dir(Path(model_dir) / task_name)

    # Download the tar file
    tar_path = task_dir / f"{task_name}.tar"
    model_url = f"{base_url}/checkpoints/{task_name}/{task_name}.tar"
    download_file(model_url, tar_path, f"Downloading {task_name} model")

    # Extract the tar file
    extract_tar(tar_path, task_dir, "Extracting model files")

    print(f"Model for {task_name} downloaded and extracted to {task_dir}")
    return task_dir


@click.command()
@click.argument(
    "task_name", type=click.Choice(["handwriting", "discrete_gestures", "wrist"])
)
@click.argument("model_dir", type=click.Path())
def main(task_name, model_dir):
    """
    Download pretrained models for neuromotor interface tasks.

    TASK_NAME: Name of the task (handwriting, discrete_gestures, or wrist)

    MODEL_DIR: Directory where the model will be stored
    """
    download_models(task_name, model_dir)


if __name__ == "__main__":
    main()
