# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, default_collate

from generic_neuromotor_interface.constants import EMG_SAMPLE_RATE
from generic_neuromotor_interface.data import (
    DataSplit,
    EmgRecording,
    HandwritingEmgDataset,
    Partitions,
    WindowedEmgDataset,
)
from generic_neuromotor_interface.utils import (
    get_full_dataset_path,
    handwriting_collate,
)


def custom_collate_fn(batch):
    """
    Custom collate function that handles pandas DataFrames and numpy arrays in batches.

    Parameters
    ----------
        batch: A list of dictionaries from __getitem__

    Returns
    -------
        A dictionary with batched tensors and non-tensor types
    """

    elem = batch[0]
    result = {}

    for key in elem:
        if isinstance(elem[key], pd.DataFrame):
            result[key] = [d[key] for d in batch]
        elif isinstance(elem[key], np.ndarray) and key == "timestamps":
            result[key] = [d[key] for d in batch]
        else:
            try:
                result[key] = default_collate([d[key] for d in batch])
            except TypeError:
                # Fallback for any other types that can't be collated
                result[key] = [d[key] for d in batch]

    return result


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        stride: int | None,
        batch_size: int,
        num_workers: int,
        data_split: DataSplit,
        transform: Callable[[EmgRecording], dict[str, torch.Tensor]],
        data_location: str,
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.stride = stride
        self.data_split = data_split

        self.batch_size = batch_size
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.num_workers = num_workers
        self.data_location = data_location

    def _make_dataset(
        self, partition_dict: dict[str, Partitions], stage: str
    ) -> ConcatDataset:

        datasets = []
        for dataset, partitions in partition_dict.items():

            # A single partition that spans the entire dataset
            if partitions is None:
                partitions = [(-np.inf, np.inf)]

            for start, end in partitions:

                # Skip partitions that are too short
                partition_samples = (end - start) * EMG_SAMPLE_RATE
                if partition_samples < self.window_length:
                    print(f"Skipping partition {dataset} {start} {end}")
                    continue

                datasets.append(
                    WindowedEmgDataset(
                        get_full_dataset_path(self.data_location, dataset),
                        start=start,
                        end=end,
                        transform=self.transform,
                        # At test time, we feed in the entire partition in one
                        # window to be more consistent with real-time deployment.
                        window_length=None if stage == "test" else self.window_length,
                        stride=None if stage == "test" else self.stride,
                        jitter=stage == "train",
                        emg_augmentation=(
                            self.emg_augmentation if stage == "train" else None
                        ),
                    )
                )
        return ConcatDataset(datasets)

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = self._make_dataset(self.data_split.train, "train")
        self.val_dataset = self._make_dataset(self.data_split.val, "val")
        self.test_dataset = self._make_dataset(self.data_split.test, "test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire partitions are
        # fed at once. Limit batch size to 1 to fit within GPU memory.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )


class HandwritingEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        padding: tuple[int, int],
        num_workers: int,
        data_split: DataSplit,
        transform: Callable[[EmgRecording], dict[str, torch.Tensor]],
        data_location: str,
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        concatenate_prompts: bool = False,
        min_duration_s: float = 0.0,
    ) -> None:
        super().__init__()
        self.collate_fn = handwriting_collate

        self.batch_size = batch_size
        self.padding = padding
        self.num_workers = num_workers
        self.data_split = data_split
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.data_location = data_location
        self.concatenate_prompts = concatenate_prompts
        self.min_duration_s = min_duration_s

    def _make_dataset(
        self, partition_dict: dict[str, Partitions], stage: str
    ) -> ConcatDataset:

        datasets = []
        for dataset, partitions in partition_dict.items():

            # A single partition that spans the entire dataset
            if partitions is None:
                partitions = [(-np.inf, np.inf)]

            for _, _ in partitions:
                datasets.append(
                    HandwritingEmgDataset(
                        get_full_dataset_path(self.data_location, dataset),
                        padding=self.padding,
                        transform=self.transform,
                        jitter=stage == "train",
                        emg_augmentation=(
                            self.emg_augmentation if stage == "train" else None
                        ),
                        concatenate_prompts=(
                            self.concatenate_prompts if stage == "train" else None
                        ),
                        min_duration_s=self.min_duration_s if stage == "train" else 0.0,
                    )
                )
        return ConcatDataset(datasets)

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = self._make_dataset(self.data_split.train, "train")
        self.val_dataset = self._make_dataset(self.data_split.val, "val")
        self.test_dataset = self._make_dataset(self.data_split.test, "test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire partitions are
        # fed at once. Limit batch size to 1 to fit within GPU memory.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
