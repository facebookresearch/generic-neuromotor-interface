# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint
from collections.abc import Callable, Sequence

from typing import Any

import hydra
import pytorch_lightning as pl
from generic_neuromotor_interface.utils import _download_data_from_aws_to

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


def train(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
    scratch_dir: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    download_subset = config.get("download_subset", False)

    # download data from S3
    dst_folder = config.data_location
    # TODO: Automatically download only the sessions in the corpus
    _download_data_from_aws_to(
        dst_folder, download_subset=download_subset, logger=logger
    )

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    logger.info("Instantiating LightningModule")
    module = instantiate(config.lightning_module, _convert_="all")

    logger.info("Instantiating LightningDataModule")
    datamodule = instantiate(config.data_module, _convert_="all")

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    # Add trainer kwargs that are dependent upon programmatic values
    # like `scratch_dir`
    other_trainer_kwargs = {}
    if scratch_dir is not None:
        other_trainer_kwargs["default_root_dir"] = scratch_dir
        logger.info(f"Setting default_root_dir to {scratch_dir=}")

    accelerator = config.trainer.get("accelerator", "auto")
    trainer = pl.Trainer(
        **config.trainer,
        **other_trainer_kwargs,
        callbacks=callbacks,
        num_nodes=config.get("num_nodes", 1),
    )

    results = {}
    if config.train:

        # Train
        logger.info("Running trainer.fit...")
        trainer.fit(module, datamodule)
        logger.info("trainer.fit completed!")

        # HACK: rank_zero_only only works for returning 1 object
        # So we return a tuple and unpack later if not None
        module_and_results = _load_and_report_best_checkpoint(
            trainer=trainer,
            module=module,
            results=results,
            logger=logger,
        )
        if module_and_results is not None:
            module, results = module_and_results

    if config.eval:

        # TODO: Set batch size to 1 for validation?
        # datamodule.batch_size = 1

        # NOTE: rank 0 only
        # Validate and test run on 1 device only (i.e. no distributed data parallelism)
        # This is to ensure reproducibility of metrics reported.
        results = _run_validate_and_test(
            module=module,
            datamodule=datamodule,
            results=results,
            logger=logger,
            accelerator=accelerator,
        )

    _log_pretty_results(results=results, logger=logger)

    return results


def evaluate_from_checkpoint(
    config: DictConfig,
    checkpoint_path: str,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # download data from S3
    dst_folder = config.data_location
    download_subset = config.get("download_subset", False)

    _download_data_from_aws_to(
        dst_folder, download_subset=download_subset, logger=logger
    )

    # Seed for determinism. This seeds torch, numpy and python random modules
    pl.seed_everything(config.seed, workers=True)

    logger.info("Instantiating (untrained) LightningModule")
    _untrained_module = instantiate(config.lightning_module, _convert_="all")

    logger.info(f"Loading checkpoint from {checkpoint_path=}...")
    module = _untrained_module.__class__.load_from_checkpoint(checkpoint_path)

    logger.info("Instantiating LightningDataModule")
    datamodule = instantiate(config.data_module, _convert_="all")

    # TODO: Conslidate/reconcile this with the logic in `train()`
    # As is, this is confusing to have 2 different trainers
    accelerator = config.trainer.get("accelerator", "auto")
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
    )

    logger.info("Running validation...")
    val_results = trainer.validate(model=module, datamodule=datamodule)
    logger.info(f"Validation completed! {val_results=}")

    logger.info("Running test...")
    test_results = trainer.test(model=module, datamodule=datamodule)
    logger.info(f"Test completed! {test_results=}")

    results = {}
    results["checkpoint_path"] = checkpoint_path
    results["val_metrics"] = val_results
    results["test_metrics"] = test_results

    return results


@pl.utilities.rank_zero_only
def _load_and_report_best_checkpoint(
    trainer, module, results, logger
) -> pl.LightningModule:

    logger.info("Loading best checkpoint...")

    checkpoint_callback = trainer.checkpoint_callback
    if checkpoint_callback is None:
        raise RuntimeError("No checkpoint callback found in trainer")

    best_checkpoint_path = checkpoint_callback.best_model_path
    best_checkpoint_score = checkpoint_callback.best_model_score

    logger.info(
        f"Loading best checkpoint from {best_checkpoint_path=} with {best_checkpoint_score=}..."
    )
    module = module.__class__.load_from_checkpoint(best_checkpoint_path)

    results["best_checkpoint_path"] = best_checkpoint_path
    results["best_checkpoint_score"] = best_checkpoint_score.cpu().item()

    return module, results


@pl.utilities.rank_zero_only
def _run_validate_and_test(module, datamodule, results, logger, accelerator):

    logger.info("Running validate and test w/ devices=1 only...")

    # TODO: Conslidate/reconcile this with the logic in `train()`
    # As is, this is confusing to have 2 different trainers
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
    )

    logger.info("Running validation...")
    val_results = trainer.validate(model=module, datamodule=datamodule)
    logger.info(f"Validation completed! {val_results=}")

    logger.info("Running test...")
    test_results = trainer.test(model=module, datamodule=datamodule)
    logger.info(f"Test completed! {test_results=}")

    results["val_metrics"] = val_results
    results["test_metrics"] = test_results

    return results


@pl.utilities.rank_zero_only
def _log_pretty_results(results, logger):
    results_formatted_str = pprint.pformat(results, sort_dicts=False)
    logger.info(f"Results: \n{results_formatted_str}")


@hydra.main(config_path="../config", config_name="wrist", version_base="1.1")
def cli(config: DictConfig):
    train(config)


if __name__ == "__main__":
    cli()
