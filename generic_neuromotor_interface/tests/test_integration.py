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

import hydra

import pytest
import pytorch_lightning as pl

from generic_neuromotor_interface.scripts.download_data import download_data
from generic_neuromotor_interface.scripts.download_models import download_models
from generic_neuromotor_interface.tests.mock_datasets import create_mock_dataset
from generic_neuromotor_interface.train import evaluate_from_checkpoint, train
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


# for now, we keep these os OS env vars for manual modification
USE_REAL_DATA = os.environ.get("USE_REAL_DATA", "False").lower() == "true"
USE_REAL_CHECKPOINTS = os.environ.get("USE_REAL_CHECKPOINTS", "False").lower() == "true"


# Define fixtures for integration tests
@pytest.fixture(scope="module")
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="module")
def temp_model_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def get_mock_datasets(task_name, temp_data_dir):
    print(f"Creating mock datasets for {task_name} in {temp_data_dir}")

    # General fixed parameters
    num_channels = 16

    # Set task-specific parameters
    if task_name == "wrist":
        # NOTE: to accommodate for wrist_mini_split.yml hard coding
        start_time = 1713966045.0
        num_samples = 200 * 2000
    else:
        start_time = 1600000000.0
        num_samples = 1000

    if task_name == "discrete_gestures":
        num_samples = 32_000  # NOTE: needed for 16_000 window size

    # Generate mock data for three users
    for user in ["000", "001", "002"]:
        _file = create_mock_dataset(
            task_name=task_name,
            output_path=temp_data_dir,
            start_time=start_time,
            num_samples=num_samples,
            num_prompts=9,
            num_channels=num_channels,
            output_file_name=f"{task_name}_user_{user}_dataset_000.hdf5",
            random_seed=0,
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
        downloaded_dir = download_data(task_name, "small_subset", temp_data_dir)
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

    # Initialize model
    pl.seed_everything(0)
    model: pl.LightningModule = hydra.utils.instantiate(
        config.lightning_module, _convert_="all"
    )

    # We need to attach the model to a trainer to call save_checkpoint
    trainer_kwargs: dict[str, Any] = hydra.utils.instantiate(
        config.trainer, _convert_="all"
    )
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
        ("discrete_gestures", "discrete_gestures"),
        ("handwriting", "handwriting"),
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
        "discrete_gestures",
        "handwriting",
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
                "trainer.accelerator=cpu",
                f"data_module/data_split={task_name}_mini_split",
            ],
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

    # Check that the best checkpoint scores
    # match expected
    key = "real_data" if USE_REAL_DATA else "mock_data"
    assert results["best_checkpoint_score"] == pytest.approx(
        REFERENCE_TRAIN_RESULTS[task_name][key]
    )


def _test_task_evaluate_mini_subset_cpu(task_name, dataset_dir, checkpoint_dir):
    """Test end-to-end inference pipeline."""
    # Test code that loads a model, runs inference on sample data,
    # and verifies the output

    assert task_name in {"discrete_gestures", "handwriting", "wrist"}

    config_dir = str(Path(__file__).parent.absolute() / "../../config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        # Compose the configuration with overrides
        base_config = compose(
            config_name=task_name,
            overrides=[
                f"data_location={str(dataset_dir)}",
                "trainer.accelerator=cpu",
                f"data_module/data_split={task_name}_mini_split",
            ],
        )

    # Load model config in checkpoint_dir and copy over interpolated values
    # from the composed base hydra config
    loaded_config = OmegaConf.load(checkpoint_dir / "model_config.yaml")
    loaded_config.data_location = base_config.data_location
    loaded_config.data_module.data_location = base_config.data_module.data_location
    loaded_config.data_module.data_split = base_config.data_module.data_split
    loaded_config.trainer.accelerator = base_config.trainer.accelerator

    assert isinstance(loaded_config, DictConfig)

    # Run training with minimal epochs
    results = evaluate_from_checkpoint(
        loaded_config, str(checkpoint_dir / "model_checkpoint.ckpt")
    )

    # Verify that training completed successfully
    assert results is not None
    assert "val_metrics" in results
    assert "test_metrics" in results

    # Check that the results match expected
    checkpoint_key = "real_checkpoint" if USE_REAL_CHECKPOINTS else "mock_checkpoint"
    data_key = "real_data" if USE_REAL_DATA else "mock_data"
    for key, metrics in results.items():
        if key == "checkpoint_path":
            continue
        for metric_name, metric_value in metrics[0].items():
            assert metric_value == pytest.approx(
                REFERENCE_EVALUATE_RESULTS[task_name][checkpoint_key][data_key][key][
                    metric_name
                ]
            )


REFERENCE_TRAIN_RESULTS = {
    "wrist": {
        "real_data": 0.09340763092041016,
        "mock_data": 0.7156573534011841,
    },
    "discrete_gestures": {
        "real_data": 0.11399950832128525,
        "mock_data": 0.1428571492433548,
    },
    "handwriting": {
        "real_data": 96.13899993896484,
        "mock_data": 98.1927719116211,
    },
}


REFERENCE_EVALUATE_RESULTS = {
    "wrist": {
        "real_checkpoint": {
            "real_data": {
                "val_metrics": {
                    "val_loss": 0.0029304532799869776,
                    "val_mae_deg_per_sec": 8.395130315569341,
                },
                "test_metrics": {
                    "test_loss": 0.0038713905960321426,
                    "test_mae_deg_per_sec": 11.0907170999639,
                },
            },
            "mock_data": {
                "val_metrics": {
                    "val_loss": 0.6564533114433289,
                    "val_mae_deg_per_sec": 1880.6002096544867,
                },
                "test_metrics": {
                    "test_loss": 0.6552098393440247,
                    "test_mae_deg_per_sec": 1877.0379244928663,
                },
            },
        },
        "mock_checkpoint": {
            "real_data": {
                "val_metrics": {
                    "val_loss": 0.053083404898643494,
                    "val_mae_deg_per_sec": 152.07275439936708,
                },
                "test_metrics": {
                    "test_loss": 0.05673551186919212,
                    "test_mae_deg_per_sec": 162.53526893095486,
                },
            },
            "mock_data": {
                "val_metrics": {
                    "val_loss": 0.6570016741752625,
                    "val_mae_deg_per_sec": 1882.1711531635895,
                },
                "test_metrics": {
                    "test_loss": 0.6555290222167969,
                    "test_mae_deg_per_sec": 1877.952316068002,
                },
            },
        },
    },
    "discrete_gestures": {
        "real_checkpoint": {
            "real_data": {
                "val_metrics": {
                    "val_accuracy": 0.556594967842102,
                    "val_loss": 0.010263126343488693,
                },
                "test_metrics": {
                    "test_cler": 0.13421496748924255,
                    "test_loss": 0.00941223930567503,
                },
            },
            "mock_data": {
                "val_metrics": {
                    "val_accuracy": 0.125,
                    "val_loss": 0.030532492324709892,
                },
                "test_metrics": {
                    "test_cler": 0.0,
                    "test_loss": 0.030042458325624466,
                },
            },
        },
        "mock_checkpoint": {
            "real_data": {
                "val_metrics": {
                    "val_accuracy": 0.1140170618891716,
                    "val_loss": 0.7204136848449707,
                },
                "test_metrics": {
                    "test_cler": 0.6722221970558167,
                    "test_loss": 0.7186295390129089,
                },
            },
            "mock_data": {
                "val_metrics": {
                    "val_loss": 0.7293607592582703,
                    "val_accuracy": 0.1428571492433548,
                },
                "test_metrics": {
                    "test_loss": 0.7633956670761108,
                    "test_cler": 0.6666666865348816,
                },
            },
        },
    },
    "handwriting": {
        "real_checkpoint": {
            "real_data": {
                "val_metrics": {
                    "val/CER": 21.750322341918945,
                    "val/DER": 4.247104167938232,
                    "val/IER": 5.53410530090332,
                    "val/SER": 11.969112396240234,
                    "val_loss": 0.6563571095466614,
                },
                "test_metrics": {
                    "test/CER": 66.04045867919922,
                    "test/DER": 0.8670520186424255,
                    "test/IER": 34.10404586791992,
                    "test/SER": 31.069364547729492,
                    "test_loss": 5.226099491119385,
                },
            },
            "mock_data": {
                "val_metrics": {
                    "val/CER": 100.0,
                    "val/DER": 0.0,
                    "val/IER": 100.0,
                    "val/SER": 0.0,
                    "val_loss": 0.0,
                },
                "test_metrics": {
                    "test/CER": 100.0,
                    "test/DER": 0.0,
                    "test/IER": 100.0,
                    "test/SER": 0.0,
                    "test_loss": 0.0,
                },
            },
        },
        "mock_checkpoint": {
            "real_data": {
                "val_metrics": {
                    "val/CER": 104.24710083007812,
                    "val/DER": 5.276705265045166,
                    "val/IER": 84.04118347167969,
                    "val/SER": 14.929214477539062,
                    "val_loss": 49.96033477783203,
                },
                "test_metrics": {
                    "test/CER": 103.68497467041016,
                    "test/DER": 4.190751552581787,
                    "test/IER": 83.16474151611328,
                    "test/SER": 16.329479217529297,
                    "test_loss": 18.270498275756836,
                },
            },
            "mock_data": {
                "val_metrics": {
                    "val/CER": 96.38554382324219,
                    "val/DER": 0.0,
                    "val/IER": 94.57831573486328,
                    "val/SER": 1.807228922843933,
                    "val_loss": 0.0,
                },
                "test_metrics": {
                    "test/CER": 96.38554382324219,
                    "test/DER": 0.0,
                    "test/IER": 94.57831573486328,
                    "test/SER": 1.807228922843933,
                    "test_loss": 0.0,
                },
            },
        },
    },
}
