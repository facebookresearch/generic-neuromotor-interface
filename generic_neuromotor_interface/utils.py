# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from hydra import compose, initialize, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn


def get_full_dataset_path(root: str, dataset: str) -> Path:
    """Add root prefix and .hdf5 suffix (if necessary) to dataset name."""
    path = Path(root).expanduser().joinpath(f"{dataset}")
    if not path.suffix:
        path = path.with_suffix(".hdf5")
    return path


def instantiate_optimizer_and_scheduler(
    params: Iterator[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig | None,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    out = {"optimizer": optimizer}

    if lr_scheduler_config is not None:
        scheduler = instantiate(lr_scheduler_config.scheduler, optimizer)
        lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
        out["lr_scheduler"] = OmegaConf.to_container(lr_scheduler)
    return out


def generate_hydra_config_from_overrides(
    config_path: str = "../config",
    version_base: str | None = None,
    config_name: str = "base",
    overrides: list[str] | None = None,
) -> DictConfig:

    if overrides is None:
        overrides = []

    if os.path.isabs(config_path):
        with initialize_config_dir(config_dir=config_path, version_base=version_base):
            config = compose(config_name=config_name, overrides=overrides)
    else:
        with initialize(config_path=config_path, version_base=version_base):
            config = compose(config_name=config_name, overrides=overrides)

    return config


def load_splits(
    metadata_file: str,
    subsample: float = 1.0,
    random_seed: int = 0,
) -> dict[str, list[str]]:
    """Load train, val, and test datasets from metadata csv."""

    # Load dataframe
    df = pd.read_csv(metadata_file)

    # Optionally subsample
    df = df.groupby("split").apply(
        lambda x: x.sample(frac=subsample, random_state=random_seed)
    )
    df.reset_index(drop=True, inplace=True)  # noqa: PD002

    # Format as dictionary
    splits = {}
    for split, df_ in df.groupby("split"):
        splits[split] = list(df_.filename)

    return splits


def _run_bash_command(bash_command, logger: logging.Logger | None = None):

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Running bash command: {bash_command}")
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)

    logger.info("Polling subprocess for stdout...")
    while True:
        std_output = process.stdout.readline()
        process_status = process.poll()
        if not std_output and process_status is not None:
            break
        if std_output:
            logger.info(std_output.strip().decode("utf-8"))
    logger.info(f"Complete! Got return code = {process_status}")


def handwriting_collate(samples: list[dict[str, torch.Tensor]]):

    emg_batch = [sample["emg"] for sample in samples if sample]  # [(T, ...)]
    prompt_batch = [sample["prompts"] for sample in samples if sample]  # [(T)]

    # Batch of inputs and targets padded along time
    padded_emg_batch = nn.utils.rnn.pad_sequence(emg_batch)  # (T, N, ...)
    padded_prompt_batch = nn.utils.rnn.pad_sequence(prompt_batch)  # (T, N)

    # Lengths of unpadded input and target sequences for each batch entry
    emg_lengths = torch.as_tensor(
        [len(_input) for _input in emg_batch], dtype=torch.int32
    )
    prompt_lengths = torch.as_tensor(
        [len(target) for target in prompt_batch], dtype=torch.int32
    )

    return {
        "emg": padded_emg_batch.movedim(0, 2),  # (T, N, ...) -> # (N, T, ...)
        "prompts": padded_prompt_batch.T,  # (T, N) -> (N, T)
        "emg_lengths": emg_lengths,
        "prompt_lengths": prompt_lengths,
    }
