# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from generic_neuromotor_interface.constants import Task
from generic_neuromotor_interface.transforms import Transform
from typing_extensions import Self


# List of start and end times within a dataset on which to train.
# Partitions correspond to times where the user is performing behaviors
# on which we want to train the model.
# If None, the entire dataset is used.
Partitions = list[tuple[float, float]] | None


@dataclass
class DataSplit:
    """Train, val, and test datasets, with partitions to sample within each dataset."""

    # {dataset_name: [(start_time, end_time), ...], ...}
    train: dict[str, Partitions | None]
    val: dict[str, Partitions | None]
    test: dict[str, Partitions | None]

    @classmethod
    def from_csv(
        cls, csv_filename: str, pool_test_partitions: bool = False
    ) -> "DataSplit":
        """Create splits from csv file with (dataset, start, end, split) columns."""

        df = pd.read_csv(csv_filename)
        splits: dict[str, dict] = {}

        for split in ["train", "val", "test"]:
            splits[split] = {}
            splits = {}
            for split in ["train", "val", "test"]:
                splits[split] = {}
                for dataset in df[df["split"] == split]["dataset"].unique():
                    dataset_rows = df[
                        (df["split"] == split) & (df["dataset"] == dataset)
                    ]

                    if split == "test" and pool_test_partitions:
                        first_start = dataset_rows["start"].min()
                        last_end = dataset_rows["end"].max()
                        splits[split][dataset] = [(first_start, last_end)]
                    else:
                        splits[split][dataset] = []
                        for row in dataset_rows.itertuples():
                            splits[split][dataset].append((row.start, row.end))

        return cls(**splits)


class EmgRecording:
    """A read-only interface to an EMG recording of a single partition, defined as a
    dataset file (HDF5 path), and a start and end time of this dataset partition.
    """

    def __init__(
        self, hdf5_path: Path, start_time: float = -np.inf, end_time: float = np.inf
    ) -> None:

        self.hdf5_path = hdf5_path
        self.start_time = start_time
        self.end_time = end_time

        # Read hdf5 timeseries data
        self._file = h5py.File(self.hdf5_path, "r")
        self.timeseries = self._file["data"]
        self.task: Task = self._file["data"].attrs["task"]

        # Prompts dataframe contains event times (no prompts for wrist datasets)
        has_prompts = "prompts" in self._file.keys()
        self.prompts = pd.read_hdf(hdf5_path, "prompts") if has_prompts else None

        # Calculate start and end indices based on timestamps
        timestamps = self.timeseries["time"]
        assert (np.diff(timestamps) >= 0).all(), "Timestamps are not monotonic"
        self.start_idx, self.end_idx = timestamps.searchsorted(
            [self.start_time, self.end_time]
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, key: slice) -> np.ndarray:
        """Slice within the recording start and stop times. The slice is relative,
        to the recording's start index.
        """

        if not isinstance(key, slice):
            raise TypeError("Only slices are supported")

        # Adjust the slice to be relative to the start index
        start = key.start if key.start is not None else 0
        stop = key.stop if key.stop is not None else len(self)
        start += self.start_idx
        stop += self.start_idx
        return self.timeseries[start:stop]

    def get_time_slice(
        self, start_t: float = -np.inf, end_t: float = np.inf
    ) -> np.ndarray:
        """Load and return a contiguous slice of the timeseries windowed
        by the provided start and end timestamps.

        Parameters
        ----------
        start_t : float, optional
            The start time of the window to grab (in absolute unix time).
            Defaults to selecting from the beginning of the session.
        end_t : float, optional
            The end time of the window to grab (in absolute unix time).
            Defaults to selecting until the end of the session.
        """
        start_idx, end_idx = self.get_idx_slice(start_t, end_t)
        return self.timeseries[start_idx:end_idx]

    def get_idx_slice(
        self, start_t: float = -np.inf, end_t: float = np.inf
    ) -> tuple[Any, Any]:
        """Load and return a contiguous slice of the timeseries windowed
        by the provided start and end timestamps.

        Parameters
        ----------
        start_t : float, optional
            The start time of the window to grab (in absolute unix time).
            Defaults to selecting from the beginning of the session.
        end_t : float, optional
            The end time of the window to grab (in absolute unix time).
            Defaults to selecting until the end of the session.
        """
        assert end_t > start_t, "start_t must be less than end_t!"
        timestamps = self.timeseries["time"]
        start_idx, end_idx = timestamps.searchsorted([start_t, end_t])
        start_idx = max(start_idx, self.start_idx)
        end_idx = min(end_idx, self.end_idx)
        return start_idx, end_idx


class WindowedEmgDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` that wraps an `EmgRecording` instance
    and provides windowing and striding functionality.

    Parameters
    ----------
    hdf5_path : Path
        Path to the HDF5 file containing the EMG recording.
    start : float
        Start time of the recording.
    end : float
        End time of the recording.
    transform : Transform
        A composed sequence of transforms that takes
        a window/slice of `EmgRecording` in the form of a numpy
        structured array and returns a `torch.Tensor` instance.
    emg_augmentation : Callable[[torch.Tensor], torch.Tensor], optional
        An optional function that takes an EMG tensor and returns
        an augmented EMG tensor. See augmentation.py.
    window_length : int, optional
        Size of each window. Specify None for no windowing, in which case
        this will be a dataset of length 1 containing the entire recording.
    stride : int, optional
        Stride between consecutive windows. Specify None to set
        this to window_length, in which case there will be no overlap
        between consecutive windows.
    jitter : bool, optional
        If True, randomly jitter the offset of each window.
        Use this for training time variability.
    """

    def __init__(
        self,
        hdf5_path: Path,
        start: float,
        end: float,
        transform: Transform,
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        window_length: int | None = 10_000,
        stride: int | None = None,
        jitter: bool = False,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.start = start
        self.end = end
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.window_length = window_length
        self.stride = stride
        self.jitter = jitter

        # Create EMG recording object
        self.emg_recording = EmgRecording(self.hdf5_path, self.start, self.end)

        # Check window length and stride
        self.window_length = (
            window_length if window_length is not None else len(self.emg_recording)
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

    def __len__(self) -> int:
        assert self.window_length is not None and self.stride is not None
        return int(
            max(len(self.emg_recording) - self.window_length, 0) // self.stride + 1
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        assert self.window_length is not None and self.stride is not None

        start_sample = idx * self.stride

        # Randomly jitter the window offset.
        leftover = len(self.emg_recording) - (start_sample + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            start_sample += np.random.randint(0, min(self.stride, leftover))

        # Get window bounds and fetch data
        window_start = max(start_sample, 0)
        window_end = start_sample + self.window_length
        timeseries = self.emg_recording[window_start:window_end]

        # Extract EMG tensor corresponding to the window
        datum: dict[str, torch.Tensor] = self.transform(
            timeseries, self.emg_recording.prompts
        )

        # Apply optional emg augmentation
        if self.emg_augmentation is not None:
            datum["emg"] = self.emg_augmentation(datum["emg"])

        # Add timestamps and prompts to the discrete_gestures test set
        # for CLER evaluation
        is_test_mode = self.window_length == len(self.emg_recording)
        is_discrete_gestures = "discrete_gestures" in str(self.hdf5_path)

        if is_test_mode and is_discrete_gestures:
            datum["prompts"] = self.emg_recording.prompts
            datum["timestamps"] = timeseries["time"]

        return datum


class HandwritingEmgDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` that wraps an `EmgRecording` instance.

    Parameters
    ----------
    hdf5_path : Path
        Path to the HDF5 file containing the EMG recording.
    padding : tuple[int, int]
        Padding to apply to the start and end of the recording.
    transform : Transform
        A composed sequence of transforms that takes
        a window/slice of `EmgRecording` in the form of a numpy
        structured array and returns a `torch.Tensor` instance.
    emg_augmentation : Callable[[torch.Tensor], torch.Tensor], optional
        An optional function that takes an EMG tensor and returns
        an augmented EMG tensor. See augmentation.py.
    jitter : bool, optional
        If True, randomly jitter the offset of each window.
        Use this for training time variability.
    concatenate_prompts : bool, optional
        If True, prompts shorter than min_duration_s will be concatenated together.
        Only prompts that follows one another will be concatenated.
    min_duration_s : float, optional
        Minimum duration of prompts to concatenate (seconds).
        Only used if concatenate_prompts is True.
    """

    def __init__(
        self,
        hdf5_path: Path,
        padding: tuple[int, int],
        transform: Transform,
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        jitter: bool = False,
        concatenate_prompts: bool = False,
        min_duration_s: float = 0.0,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.left_padding, self.right_padding = padding
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.jitter = jitter
        self.concatenate_prompts = concatenate_prompts
        self.min_duration_s = min_duration_s

        self.prompts = self.emg_recording.prompts
        self.prompts["prompt_len"] = self.prompts.end - self.prompts.start

        if self.concatenate_prompts:
            assert self.min_duration_s > 0, "min_duration_s must be positive"

            # Initialize variables
            merged_prompts = []
            current_prompt = ""
            current_len = 0
            current_start = None
            current_end = None
            # Iterate through the DataFrame
            for _, row in self.prompts.iterrows():
                if current_len == 0:
                    current_start = row["start"]

                current_prompt += row["prompt"]
                current_len += row["prompt_len"]
                current_end = row["end"]

                if current_len >= min_duration_s:
                    merged_prompts.append(
                        {
                            "prompt": current_prompt,
                            "start": current_start,
                            "end": current_end,
                            "prompt_len": current_len,
                        }
                    )
                    current_prompt = ""
                    current_len = 0
            # If there's any remaining prompt that hasn't been added
            if current_len > 0:
                merged_prompts.append(
                    {
                        "prompt": current_prompt,
                        "start": current_start,
                        "end": current_end,
                        "prompt_len": current_len,
                    }
                )
            # Create a new DataFrame with the merged prompts
            self.prompts = pd.DataFrame(merged_prompts)

    def __len__(self) -> int:
        return len(self.prompts)

    @property
    def emg_recording(self):
        if not hasattr(self, "_emg_recording"):
            self._emg_recording = EmgRecording(self.hdf5_path)
        return self._emg_recording

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor] | None:

        prompt_row = self.prompts.iloc[idx]

        start_idx, end_idx = self.emg_recording.get_idx_slice(
            start_t=prompt_row.start, end_t=prompt_row.end
        )

        timeseries = self.emg_recording[
            max(start_idx - self.left_padding, 0) : min(
                end_idx + self.right_padding, len(self.emg_recording)
            )
        ]
        if start_idx < self.left_padding or end_idx + self.right_padding > len(
            self.emg_recording
        ):
            zero_pad = (
                max(self.left_padding - start_idx, 0),
                max(end_idx + self.right_padding - len(self.emg_recording), 0),
            )
            timeseries = np.pad(timeseries, zero_pad)

        # Extract EMG tensor corresponding to the window
        datum: dict[str, torch.Tensor | str] = self.transform(
            timeseries, prompt_row.prompt
        )

        # Apply option emg augmentation
        if self.emg_augmentation is not None:
            datum["emg"] = self.emg_augmentation(datum["emg"])

        return datum
