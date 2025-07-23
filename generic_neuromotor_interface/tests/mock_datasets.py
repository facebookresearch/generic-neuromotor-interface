# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
from generic_neuromotor_interface.constants import (
    EMG_NUM_CHANNELS,
    EMG_SAMPLE_RATE,
    GestureType,
)

from generic_neuromotor_interface.data import EmgRecording


class MockEmgDatasetBase:
    """Base class for creating mock EMG datasets."""

    def __init__(
        self,
        task: str,
        num_samples: int = 1000,
        num_channels: int = EMG_NUM_CHANNELS,
        sampling_rate: float = EMG_SAMPLE_RATE,
        random_seed: Optional[int] = None,
        include_wrist_angles: bool = False,
        num_wrist_angles: int = 2,
        start_time: float | None = None,
    ) -> None:
        """
        Initialize a mock EMG dataset.

        Args:
            task: The task type (wrist, handwriting, or discrete_gestures)
            num_samples: Number of time samples to generate
            num_channels: Number of EMG channels
            sampling_rate: Sampling rate in Hz
            random_seed: Random seed for reproducibility
            include_wrist_angles: Whether to include wrist_angles field
            num_wrist_angles: Number of wrist angle dimensions
            start_time: Starting timestamp for the data
        """
        self.task = task
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.include_wrist_angles = include_wrist_angles
        self.num_wrist_angles = num_wrist_angles
        self.start_time = start_time

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        if self.start_time is None:
            self.start_time = np.random.randint(1_600_000_000, 1_700_000_000)

        # Generate timestamps
        self.timestamps = np.linspace(
            self.start_time,
            self.start_time + (num_samples - 1) / sampling_rate,
            num_samples,
        )

        # Generate mock EMG data (random values between -1 and 1)
        self.emg_data = np.random.uniform(-1, 1, (num_samples, num_channels))

        # Generate mock wrist angles if needed
        self.wrist_angles: np.ndarray | None = None
        if include_wrist_angles:
            self.wrist_angles = np.random.uniform(
                -1, 1, (num_samples, num_wrist_angles)
            )
        else:
            self.wrist_angles = None

    def create_structured_array(self) -> np.ndarray:
        """Create a structured numpy array with 'time', 'emg', and optionally
        'wrist_angles' fields.
        """
        # Create a structured array with appropriate fields
        dtype = [("time", np.float64), ("emg", np.float32, (self.num_channels,))]

        # Add wrist_angles field if needed
        if self.include_wrist_angles:
            dtype.append(("wrist_angles", np.float32, (self.num_wrist_angles,)))

        structured_array = np.zeros(self.num_samples, dtype=dtype)
        structured_array["time"] = self.timestamps
        structured_array["emg"] = self.emg_data

        if self.include_wrist_angles:
            structured_array["wrist_angles"] = self.wrist_angles

        return structured_array

    def save_to_hdf5(self, file_path: Path) -> None:
        """
        Save the mock dataset to an HDF5 file.

        Args:
            file_path: Path where the HDF5 file will be saved
        """
        # Create the directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the structured array
        data_array = self.create_structured_array()

        # Create the HDF5 file
        with h5py.File(file_path, "w") as f:
            # Create the data group and add the structured array
            data_group = f.create_dataset("data", data=data_array)

            # Set the task attribute
            data_group.attrs["task"] = self.task

            # Add prompts if needed (implemented in subclasses)
            self._add_prompts_to_hdf5(f)

    def _add_prompts_to_hdf5(self, h5file: h5py.File) -> None:
        """
        Add prompts to the HDF5 file if needed.
        This method is meant to be overridden by subclasses.

        Args:
            h5file: The open HDF5 file
        """
        pass  # No prompts for base class

    def verify_file(self, file_path: Path) -> bool:
        """
        Verify that the saved HDF5 file can be loaded by EmgRecording.

        Args:
            file_path: Path to the HDF5 file

        Returns:
            True if the file can be loaded successfully
        """
        try:
            with EmgRecording(file_path) as recording:
                # Check that the task matches
                assert (
                    recording.task == self.task
                ), f"Task mismatch: {recording.task} != {self.task}"

                # Check that the data has the expected shape
                assert (
                    len(recording) == self.num_samples
                ), f"Sample count mismatch: {len(recording)} != {self.num_samples}"

                # Check that the timestamps match
                timeseries = recording[0 : len(recording)]
                assert np.allclose(
                    timeseries["time"], self.timestamps
                ), "Timestamps don't match"

                # Check that the EMG data matches
                assert np.allclose(
                    timeseries["emg"], self.emg_data
                ), "EMG data doesn't match"

                # Check prompts if applicable (implemented in subclasses)
                self._verify_prompts(recording)

            return True
        except Exception as e:
            print(f"Verification failed: {e}")
            return False

    def _verify_prompts(self, recording: EmgRecording) -> None:
        """
        Verify that the prompts in the recording match the expected prompts.
        This method is meant to be overridden by subclasses.

        Args:
            recording: The EmgRecording instance
        """
        pass  # No prompts for base class


class MockWristEmgDataset(MockEmgDatasetBase):
    """Mock dataset for wrist task."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_channels: int = EMG_NUM_CHANNELS,
        sampling_rate: float = EMG_SAMPLE_RATE,
        random_seed: Optional[int] = None,
        num_wrist_angles: int = 2,
        start_time: float = 1600000000.0,
    ) -> None:
        """
        Initialize a mock wrist EMG dataset.

        Args:
            num_samples: Number of time samples to generate
            num_channels: Number of EMG channels
            sampling_rate: Sampling rate in Hz
            random_seed: Random seed for reproducibility
            num_wrist_angles: Number of wrist angle dimensions
        """
        super().__init__(
            task="wrist",
            num_samples=num_samples,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            random_seed=random_seed,
            include_wrist_angles=True,  # Always include wrist_angles for wrist task
            num_wrist_angles=num_wrist_angles,
            start_time=start_time,
        )


class MockDiscreteGesturesEmgDataset(MockEmgDatasetBase):
    """Mock dataset for discrete gestures task."""

    def __init__(
        self,
        num_samples: int = 32000,
        num_channels: int = EMG_NUM_CHANNELS,
        sampling_rate: float = EMG_SAMPLE_RATE,
        num_prompts: int = 5,
        gesture_labels: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        start_time: float = 1600000000.0,
    ) -> None:
        """
        Initialize a mock discrete gestures EMG dataset.

        Args:
            num_samples: Number of time samples to generate
            num_channels: Number of EMG channels
            sampling_rate: Sampling rate in Hz
            num_prompts: Number of gesture prompts to generate
            gesture_labels: List of gesture labels to use
            random_seed: Random seed for reproducibility
            start_time: Starting timestamp for the data
        """
        super().__init__(
            task="discrete_gestures",
            num_samples=num_samples,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            random_seed=random_seed,
            start_time=start_time,
        )

        self.num_prompts = num_prompts

        # Default gesture labels if none provided
        if gesture_labels is None:
            gesture_labels = [e.name for e in GestureType]
        self.gesture_labels = gesture_labels

        # Generate random prompts
        self._generate_prompts()

    def _generate_prompts(self) -> None:
        """Generate random prompts for discrete gestures."""

        prompts_data = []
        for i in range(self.num_prompts):
            # Calculate start and end times with gaps between prompts
            start_idx = int((i * 2 + 0.5) * self.num_samples / (self.num_prompts * 2))

            start_time = self.timestamps[start_idx]

            # Randomly select a gesture label
            gesture_name = np.random.choice(self.gesture_labels)

            # Create a prompt entry with the correct column names
            # The DiscreteGesturesTransform expects only 'name' and 'time' columns
            # based on the real example
            prompts_data.append(
                {
                    "name": gesture_name,  # Gesture label
                    "time": start_time,  # Event timestamp
                }
            )

        self.prompts_df = pd.DataFrame(prompts_data)

    def _add_prompts_to_hdf5(self, h5file: h5py.File) -> None:
        """
        Add prompts to the HDF5 file.

        Args:
            h5file: The open HDF5 file
        """
        # Store prompts dataframe in the HDF5 file
        self.prompts_df.to_hdf(h5file.filename, "prompts", mode="a")

    def _verify_prompts(self, recording: EmgRecording) -> None:
        """
        Verify that the prompts in the recording match the expected prompts.

        Args:
            recording: The EmgRecording instance
        """
        assert recording.prompts is not None, "Prompts should not be None"
        assert len(recording.prompts) == len(
            self.prompts_df
        ), f"Prompt count mismatch: {len(recording.prompts)} != {len(self.prompts_df)}"

        # Check that the prompts dataframe has the expected columns
        for col in ["name", "time"]:
            assert (
                col in recording.prompts.columns
            ), f"Column {col} missing from prompts"

        # Check that the prompt values match
        for i, (_, expected_row) in enumerate(self.prompts_df.iterrows()):
            actual_row = recording.prompts.iloc[i]
            assert (
                actual_row["name"] == expected_row["name"]
            ), f"Name mismatch at index {i}"
            assert np.isclose(
                actual_row["time"], expected_row["time"]
            ), f"Time mismatch at index {i}"


class MockHandwritingEmgDataset(MockEmgDatasetBase):
    """Mock dataset for handwriting task."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_channels: int = EMG_NUM_CHANNELS,
        sampling_rate: float = EMG_SAMPLE_RATE,
        num_prompts: int = 5,
        random_seed: Optional[int] = None,
        start_time: float | None = None,
    ) -> None:
        """
        Initialize a mock handwriting EMG dataset.

        Args:
            num_samples: Number of time samples to generate
            num_channels: Number of EMG channels
            sampling_rate: Sampling rate in Hz
            num_prompts: Number of text prompts to generate
            random_seed: Random seed for reproducibility
            start_time: Starting timestamp for the data
        """
        super().__init__(
            task="handwriting",
            num_samples=num_samples,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            random_seed=random_seed,
            start_time=start_time,
        )

        self.num_prompts = num_prompts

        # Generate random prompts
        self._generate_prompts()

    def _generate_prompts(self) -> None:
        """Generate random prompts for handwriting."""
        # Sample text prompts
        sample_texts = [
            "hello world",
            "neural interface",
            "machine learning",
            "artificial intelligence",
            "the quick brown fox",
            "jumped over the lazy dog",
            "meta platforms inc",
            "emg signals",
            "handwriting recognition",
        ]

        # Create random start and end times for prompts
        prompts_data = []
        for i in range(self.num_prompts):
            # Calculate start and end times with gaps between prompts
            start_idx = int((i * 2 + 0.5) * self.num_samples / (self.num_prompts * 2))
            end_idx = int((i * 2 + 1.5) * self.num_samples / (self.num_prompts * 2))

            start_time = self.timestamps[start_idx]
            end_time = self.timestamps[end_idx]

            # Randomly select a text prompt
            prompt = np.random.choice(sample_texts)

            prompts_data.append(
                {"prompt": prompt, "start": start_time, "end": end_time}
            )

        self.prompts_df = pd.DataFrame(prompts_data)

    def _add_prompts_to_hdf5(self, h5file: h5py.File) -> None:
        """
        Add prompts to the HDF5 file.

        Args:
            h5file: The open HDF5 file
        """
        # Store prompts dataframe in the HDF5 file
        self.prompts_df.to_hdf(h5file.filename, "prompts", mode="a")

    def _verify_prompts(self, recording: EmgRecording) -> None:
        """
        Verify that the prompts in the recording match the expected prompts.

        Args:
            recording: The EmgRecording instance
        """
        assert recording.prompts is not None, "Prompts should not be None"
        assert len(recording.prompts) == len(
            self.prompts_df
        ), f"Prompt count mismatch: {len(recording.prompts)} != {len(self.prompts_df)}"

        # Check that the prompts dataframe has the expected columns
        for col in ["prompt", "start", "end"]:
            assert (
                col in recording.prompts.columns
            ), f"Column {col} missing from prompts"

        # Check that the prompt values match
        for i, (_, expected_row) in enumerate(self.prompts_df.iterrows()):
            actual_row = recording.prompts.iloc[i]
            assert (
                actual_row["prompt"] == expected_row["prompt"]
            ), f"Prompt mismatch at index {i}"
            assert np.isclose(
                actual_row["start"], expected_row["start"]
            ), f"Start time mismatch at index {i}"
            assert np.isclose(
                actual_row["end"], expected_row["end"]
            ), f"End time mismatch at index {i}"


def create_mock_dataset(
    task_name: str,
    output_path: Path,
    num_samples: int = 1000,
    num_channels: int = EMG_NUM_CHANNELS,
    sampling_rate: float = EMG_SAMPLE_RATE,
    num_prompts: int = 5,
    random_seed: Optional[int] = None,
    num_wrist_angles: int = 2,
    output_file_name: Optional[str] = None,
    start_time: float = 1600000000.0,
) -> Path:
    """
    Create a mock dataset for the specified task and save it to an HDF5 file.

    Args:
        task_name: Name of the task ('wrist', 'handwriting', or 'discrete_gestures')
        output_path: Directory where the HDF5 file will be saved
        num_samples: Number of time samples to generate
        num_channels: Number of EMG channels
        sampling_rate: Sampling rate in Hz
        num_prompts: Number of prompts to generate
            (for handwriting and discrete_gestures)
        random_seed: Random seed for reproducibility
        num_wrist_angles: Number of wrist angle dimensions (for wrist task)
        output_file_name: Optional custom filename for the output file
        start_time: Starting timestamp for the data

    Returns:
        Path to the created HDF5 file
    """
    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Create the appropriate mock dataset based on the task
    dataset: MockEmgDatasetBase
    if task_name == "wrist":
        dataset = MockWristEmgDataset(
            num_samples=num_samples,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            random_seed=random_seed,
            num_wrist_angles=num_wrist_angles,
            start_time=start_time,
        )
    elif task_name == "handwriting":
        dataset = MockHandwritingEmgDataset(
            num_samples=num_samples,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            num_prompts=num_prompts,
            random_seed=random_seed,
            start_time=start_time,
        )
    elif task_name == "discrete_gestures":
        dataset = MockDiscreteGesturesEmgDataset(
            num_samples=num_samples,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            num_prompts=num_prompts,
            random_seed=random_seed,
            start_time=start_time,
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Create the file path
    if output_file_name is None:
        output_file_name = f"{task_name}_mock.hdf5"
    file_path = output_path / output_file_name

    # Save the dataset to an HDF5 file
    dataset.save_to_hdf5(file_path)

    # Verify the file
    if not dataset.verify_file(file_path):
        raise RuntimeError(f"Failed to verify the created file: {file_path}")

    return file_path
