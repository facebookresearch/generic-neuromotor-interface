# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

from generic_neuromotor_interface.data import EmgRecording
from generic_neuromotor_interface.tests.mock_datasets import (
    create_mock_dataset,
    MockDiscreteGesturesEmgDataset,
    MockHandwritingEmgDataset,
    MockWristEmgDataset,
)


def test_mock_wrist_dataset():
    """Test creating and loading a mock wrist dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock wrist dataset
        num_wrist_angles = 2
        dataset = MockWristEmgDataset(
            num_samples=500,
            num_channels=8,
            sampling_rate=1000.0,
            random_seed=42,
            num_wrist_angles=num_wrist_angles,
        )

        # Save it to an HDF5 file
        file_path = temp_path / "wrist_test.h5"
        dataset.save_to_hdf5(file_path)

        # Verify that the file can be loaded by EmgRecording
        with EmgRecording(file_path) as recording:
            # Check that the task is correct
            assert recording.task == "wrist"

            # Check that the data has the expected shape
            assert len(recording) == 500

            # Check that the prompts are None (wrist datasets don't have prompts)
            assert recording.prompts is None

            # Load a slice of the data
            data_slice = recording[0:100]
            assert data_slice.shape == (100,)
            assert "time" in data_slice.dtype.names
            assert "emg" in data_slice.dtype.names
            assert "wrist_angles" in data_slice.dtype.names
            assert data_slice["emg"].shape == (100, 8)
            assert data_slice["wrist_angles"].shape == (100, num_wrist_angles)


def test_mock_discrete_gestures_dataset():
    """Test creating and loading a mock discrete gestures dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock discrete gestures dataset
        dataset = MockDiscreteGesturesEmgDataset(
            num_samples=500,
            num_channels=8,
            sampling_rate=1000.0,
            num_prompts=3,
            gesture_labels=["pinch", "point", "fist"],
            random_seed=42,
        )

        # Save it to an HDF5 file
        file_path = temp_path / "discrete_gestures_test.h5"
        dataset.save_to_hdf5(file_path)

        # Verify that the file can be loaded by EmgRecording
        with EmgRecording(file_path) as recording:
            # Check that the task is correct
            assert recording.task == "discrete_gestures"

            # Check that the data has the expected shape
            assert len(recording) == 500

            # Check that the prompts dataframe exists and has the expected structure
            assert recording.prompts is not None
            assert len(recording.prompts) == 3
            assert "name" in recording.prompts.columns
            assert "time" in recording.prompts.columns
            assert len(recording.prompts.columns) == 2  # Only name and time columns

            # Check that all prompts are from the provided labels
            for gesture_name in recording.prompts["name"]:
                assert gesture_name in ["pinch", "point", "fist"]


def test_mock_handwriting_dataset():
    """Test creating and loading a mock handwriting dataset."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a mock handwriting dataset
        dataset = MockHandwritingEmgDataset(
            num_samples=500,
            num_channels=8,
            sampling_rate=1000.0,
            num_prompts=3,
            random_seed=42,
        )

        # Save it to an HDF5 file
        file_path = temp_path / "handwriting_test.h5"
        dataset.save_to_hdf5(file_path)

        # Verify that the file can be loaded by EmgRecording
        with EmgRecording(file_path) as recording:
            # Check that the task is correct
            assert recording.task == "handwriting"

            # Check that the data has the expected shape
            assert len(recording) == 500

            # Check that the prompts dataframe exists and has the expected structure
            assert recording.prompts is not None
            assert len(recording.prompts) == 3
            assert "prompt" in recording.prompts.columns
            assert "start" in recording.prompts.columns
            assert "end" in recording.prompts.columns

            # Check that all prompts are strings
            for prompt in recording.prompts["prompt"]:
                assert isinstance(prompt, str)


def test_create_mock_dataset_utility():
    """Test the create_mock_dataset utility function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock datasets for all three tasks
        tasks = ["wrist", "handwriting", "discrete_gestures"]

        for task in tasks:
            # Create a mock dataset
            file_path = create_mock_dataset(
                task_name=task,
                output_path=temp_path,
                num_samples=200,
                num_channels=8,
                sampling_rate=1000.0,
                num_prompts=2,
                random_seed=42,
            )

            # Check that the file exists
            assert file_path.exists()

            # Verify that the file can be loaded by EmgRecording
            with EmgRecording(file_path) as recording:
                # Check that the task is correct
                assert recording.task == task

                # Check that the data has the expected shape
                assert len(recording) == 200

                # Check prompts based on task
                if task == "wrist":
                    assert recording.prompts is None
                else:
                    assert recording.prompts is not None
                    assert len(recording.prompts) == 2
