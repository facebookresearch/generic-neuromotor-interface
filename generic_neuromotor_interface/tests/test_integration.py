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

import os
import tempfile
from pathlib import Path
from typing import Any
import numpy as np

import hydra

import pytest
import pytorch_lightning as pl

from distutils.util import strtobool

from generic_neuromotor_interface.scripts.download_data import download_data
from generic_neuromotor_interface.scripts.download_models import download_models
from generic_neuromotor_interface.tests.mock_datasets import create_mock_dataset
from generic_neuromotor_interface.train import evaluate_from_checkpoint, train
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


# for now, we keep these os OS env vars for manual modification
USE_REAL_DATA = strtobool(os.environ.get("USE_REAL_DATA", "False"))
USE_FULL_DATA = strtobool(os.environ.get("USE_FULL_DATA", "False"))

if USE_FULL_DATA and not USE_REAL_DATA:
    raise ValueError(
        f"Got {USE_FULL_DATA=} but {USE_REAL_DATA=}."
        f"{USE_FULL_DATA=} should only be used with USE_REAL_DATA=True"
    )

USE_PERSISTENT_TEMP_DIR = strtobool(os.environ.get("USE_PERSISTENT_TEMP_DIR", "True"))

USE_REAL_CHECKPOINTS = strtobool(os.environ.get("USE_REAL_CHECKPOINTS", "False"))
USE_CUDA = strtobool(os.environ.get("USE_CUDA", "False"))

print(
    f"Using configuration: {USE_REAL_DATA=} {USE_FULL_DATA=} {USE_REAL_CHECKPOINTS=} {USE_CUDA=}"
)


# Define fixtures for integration tests
@pytest.fixture(scope="module")
def temp_data_dir():
    """Create a temporary directory for test data."""
    if USE_PERSISTENT_TEMP_DIR and USE_REAL_DATA:
        path = Path(tempfile.gettempdir()) / "emg_test_data_cache"
        print(f"Using persistent temp dir at: {path=}")
        yield path
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


@pytest.fixture(scope="module")
def temp_model_dir():
    """Create a temporary directory for test data."""
    if USE_PERSISTENT_TEMP_DIR and USE_REAL_DATA:
        path = Path(tempfile.gettempdir()) / "emg_test_data_cache"
        print(f"Using persistent temp dir at: {path=}")
        yield path
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


def get_mock_datasets(task_name, temp_data_dir):
    print(f"Creating mock datasets for {task_name} in {temp_data_dir}")
    for user in ["000", "001", "002"]:

        num_samples = 32_000  # NOTE: needed for 16_000 window size discrete_gestures

        _file = create_mock_dataset(
            task_name=task_name,
            output_path=temp_data_dir,
            num_samples=num_samples,
            num_prompts=9,
            output_file_name=f"{task_name}_user_{user}_dataset_000.hdf5",
        )
        assert _file is not None
        assert _file.exists()
        print(f"Created {_file}")

    return temp_data_dir


@pytest.fixture(scope="module")
def task_dataset_dir_fixture(request, temp_data_dir):
    """Create a sample dataset for integration tests."""

    task_name = request.param

    if USE_REAL_DATA:
        dataset_subset = "small_subset" if not USE_FULL_DATA else "full_data"
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
@pytest.fixture(scope="module")
def task_model_fixture(request, temp_model_dir):
    """
    Fixture that provides task-specific model data for evaluation tests.

    The request.param should be the task name
    (e.g., 'wrist', 'handwriting', 'discrete_gestures').
    """
    task_name = request.param

    if USE_REAL_CHECKPOINTS:
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
    "task_model_fixture,task_dataset_dir_fixture",
    [
        ("wrist", "wrist"),
        ("handwriting", "handwriting"),
        ("discrete_gestures", "discrete_gestures"),
    ],
    indirect=True,
)
def test_task_evaluate_subset_cpu(task_model_fixture, task_dataset_dir_fixture):
    """
    Test task evaluation using indirect parameterization.
    """
    task_name = task_model_fixture["task_name"]
    model_dir = task_model_fixture["model_dir"]
    dataset_dir = task_dataset_dir_fixture["dataset_dir"]
    assert task_name == task_dataset_dir_fixture["task_name"]

    _test_task_evaluate_mini_subset_cpu(task_name, dataset_dir, model_dir)


@pytest.mark.integration
@pytest.mark.parametrize(
    "task_dataset_dir_fixture",
    [
        "wrist",
        "handwriting",
        "discrete_gestures",
    ],
    indirect=True,
)
def test_task_train_subset_cpu(task_dataset_dir_fixture):
    """
    Test task training using indirect parameterization.
    """
    task_name = task_dataset_dir_fixture["task_name"]
    data_dir = task_dataset_dir_fixture["dataset_dir"]

    _test_task_train_mini_subset_cpu(task_name, data_dir)


def _test_task_train_mini_subset_cpu(task_name, dataset_dir):
    """Test the model training pipeline for a specific task."""

    config_dir = str(Path(__file__).parent.absolute() / "../../config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        # Compose the configuration with overrides
        config = compose(
            config_name=task_name,
            overrides=[
                f"data_location={str(dataset_dir)}",
                "trainer.max_epochs=1",
                f"trainer.accelerator={'cpu' if not USE_CUDA else 'cuda'}",
            ]
            + (
                [f"data_module/data_split={task_name}_mini_split"]
                if not USE_FULL_DATA
                else []
            ),
        )

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


def _check_expected_results(task_name: str, results: dict[str, Any]):
    if USE_FULL_DATA:
        if task_name == "wrist":
            _assert_expected(
                actual=results["test_metrics"][0]["test_mae_deg_per_sec"],
                expected=11.2348,
                metric_name="wrist:test_mae_deg_per_sec",
            )
        elif task_name == "discrete_gestures":
            _assert_expected(
                actual=results["test_metrics"][0]["test_cler"],
                expected=0.1819,
                metric_name="discrete_gestures:test_cler",
            )
        elif task_name == "handwriting":
            _assert_expected(
                actual=results["test_metrics"][0]["test/CER"],
                expected=30.0686,
                metric_name="handwriting:test/CER",
            )
        else:
            raise ValueError(f"Unrecogznied {task_name=}")


def _test_task_evaluate_mini_subset_cpu(task_name, dataset_dir, checkpoint_dir):
    """Test end-to-end inference pipeline."""
    # Test code that loads a model, runs inference on sample data,
    # and verifies the output

    assert task_name in {"discrete_gestures", "handwriting", "wrist"}
    print(f"Running evaluation test for {task_name=} {dataset_dir=} {checkpoint_dir=}")

    config_dir = str(Path(__file__).parent.absolute() / "../../config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        # Compose the configuration with overrides
        base_config = compose(
            config_name=task_name,
            overrides=[
                f"data_location={str(dataset_dir)}",
                f"trainer.accelerator={'cpu' if not USE_CUDA else 'cuda'}",
            ]
            + (
                [f"data_module/data_split={task_name}_mini_split"]
                if not USE_FULL_DATA
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
        evaluate_validation_set = False  # we can skip val since it's tested during other tests

        results = evaluate_from_checkpoint(
            loaded_config, str(checkpoint_dir / "model_checkpoint.ckpt"), evaluate_validation_set=evaluate_validation_set
        )

        # Verify that training completed successfully
        assert results is not None

        if evaluate_validation_set:
            assert "val_metrics" in results

        assert "test_metrics" in results

        _check_expected_results(task_name, results)
