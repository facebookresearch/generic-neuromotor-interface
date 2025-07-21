# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests.

These tests are marked with @pytest.mark.integration and are excluded by default.

They can be explicitly run with:
    pytest -m integration

Regular tests can be run with:
    pytest
"""

import tempfile

from pathlib import Path
from typing import Any

import hydra
import numpy as np

import pytest
import pytorch_lightning as pl
import torch
from generic_neuromotor_interface.constants import EMG_SAMPLE_RATE

from generic_neuromotor_interface.scripts.download_data import download_data
from generic_neuromotor_interface.scripts.download_models import download_models
from generic_neuromotor_interface.tests.expected_test_results import (
    EXPECTED_TEST_VALUES,
)
from generic_neuromotor_interface.tests.mock_datasets import create_mock_dataset
from generic_neuromotor_interface.train import evaluate_from_checkpoint, train
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


# Define fixtures for integration tests
@pytest.fixture(scope="function")
def create_temp_dir(use_persistent_temp_dir, use_real_data):
    """Create a temporary directory based on parameters."""
    if use_persistent_temp_dir and use_real_data:
        path = Path(tempfile.gettempdir()) / "emg_test_data_cache"
        print(f"Using persistent temp dir at: {path=}")
        yield path
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


def get_mock_datasets(task_name, temp_data_dir):
    print(f"Creating mock datasets for {task_name} in {temp_data_dir}")
    config_dir = str(Path(__file__).parent.absolute() / "../../config")

    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        config = compose(
            config_name=task_name,
            overrides=[f"data_module/data_split={task_name}_mini_split"],
        )

    data_split = hydra.utils.instantiate(config.data_module.data_split)

    for split in ["train", "val", "test"]:
        data_in_split = getattr(data_split, split)
        for dataset_name, partitions in data_in_split.items():
            if partitions is None:
                _file = create_mock_dataset(
                    task_name=task_name,
                    output_path=temp_data_dir,
                    num_samples=32_000,
                    num_prompts=9,
                    output_file_name=f"{dataset_name}.hdf5",
                )
                continue

            for partition in partitions:
                start, end = partition
                _buffer = 5.0  # add a time buffer around selected window (start, end)
                _file = create_mock_dataset(
                    task_name=task_name,
                    output_path=temp_data_dir,
                    start_time=start - _buffer,
                    num_samples=int((end - start + _buffer) * EMG_SAMPLE_RATE),
                    num_prompts=9,
                    output_file_name=f"{dataset_name}.hdf5",
                )
                assert _file is not None
                assert _file.exists()
                print(f"Created {_file}")

    return temp_data_dir


@pytest.fixture(scope="function")
def task_dataset_dir_fixture(task_name, use_real_data, use_full_data, create_temp_dir):
    """Create a sample dataset for integration tests."""
    temp_data_dir = create_temp_dir

    # Validate configuration
    if use_full_data and not use_real_data:
        pytest.skip(
            f"Invalid configuration: use_full_data={use_full_data} but "
            f"use_real_data={use_real_data}. "
            "use_full_data should only be used with use_real_data=True"
        )

    if use_real_data:
        dataset_subset = "small_subset" if not use_full_data else "full_data"
        downloaded_dir = download_data(task_name, dataset_subset, temp_data_dir)
        assert downloaded_dir is not None
        assert downloaded_dir.exists()
    else:
        downloaded_dir = get_mock_datasets(task_name, temp_data_dir)

    # For now, just return the directory
    return {
        "task_name": task_name,
        "dataset_dir": downloaded_dir,
    }


def get_mock_checkpoint_dir(task_name, temp_model_dir):
    print(f"Creating mock checkpoint for {task_name} in {temp_model_dir}")
    # Create a mock checkpoint directory
    model_dir = temp_model_dir / task_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create a mock model checkpoint file
    config_dir = str(Path(__file__).parent.absolute() / "../../config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        # Compose the configuration with overrides
        config = compose(
            config_name=task_name,
        )
        model: pl.LightningModule = hydra.utils.instantiate(
            config.lightning_module, _convert_="all"
        )
        trainer_kwargs: dict[str, Any] = hydra.utils.instantiate(
            config.trainer, _convert_="all"
        )
        # We need to attach the model to a trainer to call save_checkpoint
        trainer = pl.Trainer(**trainer_kwargs)
        try:
            trainer.fit(model)
        except ValueError:
            pass
        trainer.save_checkpoint(model_dir / "model_checkpoint.ckpt")

    # Dump the config to a yaml file
    with open(model_dir / "model_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    return model_dir


# Fixture for evaluation tests
@pytest.fixture(scope="function")
def task_model_fixture(task_name, use_real_checkpoints, create_temp_dir):
    """
    Fixture that provides task-specific model data for evaluation tests.
    """
    temp_model_dir = create_temp_dir

    if use_real_checkpoints:
        # Download the model checkpoint
        model_dir = download_models(task_name, temp_model_dir)
    else:
        model_dir = get_mock_checkpoint_dir(task_name, temp_model_dir)
    assert model_dir is not None
    assert model_dir.exists()
    assert (model_dir / "model_checkpoint.ckpt").exists()
    assert (model_dir / "model_config.yaml").exists()

    return {
        "task_name": task_name,
        "model_dir": model_dir,
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    "use_real_data",
    [
        False,
        pytest.param(True, marks=pytest.mark.real_data),
    ],
    ids=lambda val: f"real_data={val}",
)
@pytest.mark.parametrize(
    "use_full_data",
    [
        False,
        pytest.param(True, marks=pytest.mark.full_data),
    ],
    ids=lambda val: f"full_data={val}",
)
@pytest.mark.parametrize(
    "use_persistent_temp_dir",
    [
        True,
    ],
    ids=lambda val: f"persistent_temp_dir={val}",
)
@pytest.mark.parametrize(
    "use_real_checkpoints",
    [
        False,
        pytest.param(True, marks=pytest.mark.real_checkpoints),
    ],
    ids=lambda val: f"real_checkpoints={val}",
)
@pytest.mark.parametrize(
    "use_cuda",
    [
        False,
        pytest.param(True, marks=pytest.mark.cuda),
    ],
    ids=lambda val: f"cuda={val}",
)
@pytest.mark.parametrize(
    "task_name",
    [
        "wrist",
        "handwriting",
        "discrete_gestures",
    ],
    ids=lambda val: f"task={val}",
)
def test_task_evaluate_from_checkpoint(
    use_real_data,
    use_full_data,
    use_persistent_temp_dir,
    use_real_checkpoints,
    use_cuda,
    task_name,
    task_model_fixture,
    task_dataset_dir_fixture,
):
    """
    Test task evaluation using direct parameterization.
    """
    # Skip invalid combinations
    if use_full_data and not use_real_data:
        pytest.skip("use_full_data=True requires use_real_data=True")

    if use_real_data and not use_real_checkpoints:
        pytest.skip("skipping fake checkpoints + real data combo; redundant")

    model_dir = task_model_fixture["model_dir"]
    dataset_dir = task_dataset_dir_fixture["dataset_dir"]

    print(
        f"Using configuration: use_real_data={use_real_data} "
        f"use_full_data={use_full_data} "
        f"use_real_checkpoints={use_real_checkpoints} "
        f"use_cuda={use_cuda}"
    )

    if use_cuda:
        print("Clearing CUDA cache [start]")
        torch.cuda.empty_cache()

    _test_task_evaluate_mini_subset_cpu(
        task_name,
        dataset_dir,
        model_dir,
        use_full_data=use_full_data,
        use_cuda=use_cuda,
        use_real_checkpoints=use_real_checkpoints,
    )

    if use_cuda:
        print("Clearing CUDA cache [end]")
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.parametrize(
    "use_real_data",
    [
        False,
        pytest.param(True, marks=pytest.mark.real_data),
    ],
    ids=lambda val: f"real_data={val}",
)
@pytest.mark.parametrize(
    "use_full_data",
    [
        False,
        pytest.param(True, marks=pytest.mark.full_data),
    ],
    ids=lambda val: f"full_data={val}",
)
@pytest.mark.parametrize(
    "use_persistent_temp_dir",
    [
        True,
    ],
    ids=lambda val: f"persistent_temp_dir={val}",
)
@pytest.mark.parametrize(
    "use_cuda",
    [
        False,
        pytest.param(True, marks=pytest.mark.cuda),
    ],
    ids=lambda val: f"cuda={val}",
)
@pytest.mark.parametrize(
    "task_name",
    [
        "wrist",
        "handwriting",
        "discrete_gestures",
    ],
    ids=lambda val: f"task={val}",
)
def test_task_training_loop(
    use_real_data,
    use_full_data,
    use_persistent_temp_dir,
    use_cuda,
    task_name,
    task_dataset_dir_fixture,
):
    """
    Test task training using direct parameterization.
    """
    # Skip invalid combinations
    if use_full_data and not use_real_data:
        pytest.skip("use_full_data=True requires use_real_data=True")

    data_dir = task_dataset_dir_fixture["dataset_dir"]

    print(
        f"Using configuration: use_real_data={use_real_data} "
        f"use_full_data={use_full_data} "
        f"use_cuda={use_cuda}"
    )

    if use_cuda:
        print("Clearing CUDA cache [start]")
        torch.cuda.empty_cache()

    _test_task_train_mini_subset_cpu(
        task_name,
        data_dir,
        use_full_data=use_full_data,
        use_cuda=use_cuda,
    )

    if use_cuda:
        print("Clearing CUDA cache [end]")
        torch.cuda.empty_cache()


def _test_task_train_mini_subset_cpu(
    task_name, dataset_dir, use_full_data=False, use_cuda=False
):
    """Test the model training pipeline for a specific task."""
    print(
        f"Running training test for {task_name=} {dataset_dir=} "
        f"{use_full_data=} {use_cuda=}"
    )

    config_dir = str(Path(__file__).parent.absolute() / "../../config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        # Compose the configuration with overrides
        config = compose(
            config_name=task_name,
            overrides=[
                f"data_location={str(dataset_dir)}",
                "trainer.max_epochs=1",
                f"trainer.accelerator={'cpu' if not use_cuda else 'cuda'}",
            ]
            + (
                [f"data_module/data_split={task_name}_mini_split"]
                if not use_full_data
                else []
            ),
        )

        # Reduce batch size for tasks to accomodate GitHub CI GPU runners
        if use_cuda:
            if task_name == "wrist":
                config.data_module.batch_size = 8  # from 256
            elif task_name == "handwriting":
                config.data_module.batch_size = 2  # from 8

        # Run training with minimal epochs
        results = train(config)

        # Verify that training completed successfully
        assert results is not None
        assert "best_checkpoint_path" in results
        assert "best_checkpoint_score" in results

        if config.eval:
            assert "val_metrics" in results
            assert "test_metrics" in results


def _assert_expected(actual: float, expected: float, metric_name: str, atol=1e-3):
    delta = actual - expected
    print(f"[{metric_name}] Got {actual=}. Expected {expected=}. Delta {delta=}")
    np.testing.assert_allclose(actual, expected, atol=atol)


def _check_expected_results(
    task_name: str,
    results: dict[str, Any],
    use_full_data=False,
    use_real_checkpoints=False,
):

    # Only check if using real checkpoints
    if not use_real_checkpoints:
        return

    # Determine which dataset size we're using
    data_size = "full_data" if use_full_data else "small_subset"

    # Check if task exists in our expectations
    if task_name not in EXPECTED_TEST_VALUES:
        raise ValueError(f"Unrecognized {task_name=}")

    # Check if we have expectations for this data size
    if data_size not in EXPECTED_TEST_VALUES[task_name]:
        return

    # Check each expected metric
    for metric_name, expectation in EXPECTED_TEST_VALUES[task_name][data_size].items():
        _assert_expected(
            actual=results["test_metrics"][0][metric_name],
            expected=expectation["value"],
            metric_name=f"{task_name}:{metric_name}",
            atol=expectation["atol"],
        )


def _test_task_evaluate_mini_subset_cpu(
    task_name,
    dataset_dir,
    checkpoint_dir,
    use_full_data=False,
    use_cuda=False,
    use_real_checkpoints=False,
):
    """Test end-to-end inference pipeline."""
    # Test code that loads a model, runs inference on sample data,
    # and verifies the output

    assert task_name in {"discrete_gestures", "handwriting", "wrist"}
    print(
        f"Running evaluation test for {task_name=} {dataset_dir=} "
        f"{checkpoint_dir=} {use_full_data=} {use_cuda=} {use_real_checkpoints=}"
    )

    config_dir = str(Path(__file__).parent.absolute() / "../../config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        # Compose the configuration with overrides
        base_config = compose(
            config_name=task_name,
            overrides=[
                f"data_location={str(dataset_dir)}",
                f"trainer.accelerator={'cpu' if not use_cuda else 'cuda'}",
            ]
            + (
                [f"data_module/data_split={task_name}_mini_split"]
                if not use_full_data
                else []
            ),
        )

        loaded_config = OmegaConf.load(checkpoint_dir / "model_config.yaml")
        loaded_config.data_location = base_config.data_location
        loaded_config.data_module.data_location = base_config.data_module.data_location
        loaded_config.data_module.data_split = base_config.data_module.data_split
        loaded_config.trainer.accelerator = base_config.trainer.accelerator

        assert isinstance(loaded_config, DictConfig)
        print(OmegaConf.to_yaml(loaded_config))

        # Run eval
        evaluate_validation_set = (
            False  # we can skip val since it's tested during other tests
        )

        results = evaluate_from_checkpoint(
            loaded_config,
            str(checkpoint_dir / "model_checkpoint.ckpt"),
            evaluate_validation_set=evaluate_validation_set,
        )

        # Verify that training completed successfully
        assert results is not None

        if evaluate_validation_set:
            assert "val_metrics" in results

        assert "test_metrics" in results

        _check_expected_results(task_name, results, use_full_data, use_real_checkpoints)
