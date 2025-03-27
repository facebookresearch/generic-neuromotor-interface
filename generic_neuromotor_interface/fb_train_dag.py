# TODO: REMOVE ME FOR OSS

# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import pprint
from typing import Any

import attr

import kfp.dsl as kfp_dsl

from ctrl.logging import get_logger

from generic_neuromotor_interface.train import evaluate_from_checkpoint, train
from memo.cache.base import unique_id

from omegaconf import DictConfig, OmegaConf
from pybmi.modeling.rp_components.base_model import RPModel
from pybmi.torch.callbacks import ReportCallback
from workflow.components import ctrl_dsl
from workflow.components.base import ctrl_comp, MachineType
from workflow.components.rp_components import report_model
from workflow.dags.base import DagBase
from workflow.dags.context import start_run
from workflow.dags.utils import create_ddp_cluster

# TODO: Move this script out of the open source repo
RP_MODEL_CLASS = "oss_nature_paper"


def train_component_fn(config: dict[str, Any], num_nodes: int = 1, node_rank: int = 0):

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger = get_logger(__name__ + " " + f"[{node_rank=}][{local_rank=}]")

    manager = start_run(profile=False)
    model_id = unique_id(prefix="pytorch")
    name = model_id if "dag" not in config else config["dag"]["name"]

    with manager as ctx:

        is_rank_zero, __ = create_ddp_cluster(
            ctx, node_rank=node_rank, num_nodes=num_nodes
        )
        if is_rank_zero and ctx is not None:
            ctx.set_property("master_model_id", model_id)

        with report_model(
            ctx=ctx if is_rank_zero else None,
            config=config,
            model_class=RP_MODEL_CLASS,
            id=model_id,
            name=name,
        ) as rp_reporter:

            if isinstance(config, dict):
                config = OmegaConf.create(config)

            progress_fn = rp_reporter.get_report_progress_fn()
            extra_callbacks = [ReportCallback(progress_fn, metric_name="val_loss")]

            # IDEA: modify / inject info into the config to add RP compatibility
            results = train(
                config=config,
                scratch_dir=rp_reporter.scratch_dir,
                extra_callbacks=extra_callbacks,  # pyre-ignore[6]
                logger=logger,
            )

            # report results to RP (rank 0 only)
            if is_rank_zero:

                final_s3_uri = None
                if "best_checkpoint_path" in results:
                    ckpt_path = results["best_checkpoint_path"]
                    rel_path = os.path.relpath(ckpt_path, rp_reporter.scratch_dir)

                    if rp_reporter.get_report_progress_fn().use_external_services:
                        final_s3_uri = os.path.join(rp_reporter.s3_uri, rel_path)
                    else:
                        final_s3_uri = ckpt_path

                    results["best_checkpoint_path_s3"] = final_s3_uri

                if ctx is not None:
                    logger.info(
                        f"Setting best checkpoint path ({final_s3_uri=}) in RP context..."
                    )
                    ctx.set_property("best_checkpoint_path_s3", final_s3_uri)

                top_level_score = None
                if "best_checkpoint_score" in results:
                    top_level_score = results["best_checkpoint_score"]

                logger.info("Uploading and reporting model to RP...")
                rp_reporter.upload_and_report_model(
                    final_s3_uri=final_s3_uri,
                    top_level_score=top_level_score,
                    results_json=json.dumps(results),
                )
                logger.info("Done uploading and reporting model to RP!")

                results_formatted_str = pprint.pformat(results, sort_dicts=False)
                logger.info(f"FINAL RESULTS: \n{results_formatted_str}")

            # TODO: Necessary?
            if ctx is not None:
                model_id = str(ctx.get_property("master_model_id"))

            return model_id


# shared eval code across different DAG component entrypoints
def _evaluate(ctx, config, checkpoint_path_uri, logger, model_id, name):
    with report_model(
        ctx=ctx,
        config=config,
        model_class=RP_MODEL_CLASS,
        id=model_id,
        name=name,
    ) as rp_reporter:

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        results = evaluate_from_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path_uri,
            logger=logger,
        )

        top_level_score = None
        if "val_metrics" in results:
            try:
                top_level_score = results["val_metrics"][0]["val_loss"]
            except (KeyError, IndexError):
                logger.error("Could not find top level score in val_metrics!")
                top_level_score = None

        rp_reporter.upload_and_report_model(
            final_s3_uri=checkpoint_path_uri,
            top_level_score=top_level_score,
            results_json=json.dumps(results),
        )

        results_formatted_str = pprint.pformat(results, sort_dicts=False)
        logger.info(f"FINAL RESULTS: \n{results_formatted_str}")

        return model_id


# component entrypoint for for eval after training
# uses the same config as used for training; uses DAG ctx to fetch best checkpoint path, model_id
def evaluate_after_train_fn(
    config: dict[str, Any], checkpoint_path_uri: str | None = None
):

    logger = get_logger(__name__)
    manager = start_run(profile=False)

    with manager as ctx:

        if checkpoint_path_uri is None and ctx is not None:
            logger.info("Fetching best checkpoint path from RP context...")
            checkpoint_path_uri = str(ctx.get_property("best_checkpoint_path_s3"))
        logger.info(f"Best checkpoint path: {checkpoint_path_uri}")

        if ctx is not None:
            train_model_id = str(ctx.get_property("master_model_id"))
            model_id = "EVALULATE_" + train_model_id
            name = model_id if "dag" not in config else config["dag"]["name"]

            model_id = _evaluate(
                ctx=ctx,
                config=config,
                checkpoint_path_uri=checkpoint_path_uri,
                logger=logger,
                model_id=model_id,
                name=name,
            )
        return model_id


# component entrypoint for eval from just model_id
# infers config from RP, infers best checkpoint path from RP (if not provided)
# TODO: consider using this as the default entrypoint for eval after train too?
def evaluate_from_model_id(model_id: str, checkpoint_path_uri: str | None = None):

    logger = get_logger(__name__)
    manager = start_run(profile=False)

    # get config from RP
    model_info = RPModel.get_model_info_from_id(model_id=model_id)
    config = model_info["config"]
    logger.info(f"Found config from {model_id=}: {config}")

    # get best checkpoint path from results
    if checkpoint_path_uri is None:
        checkpoint_path_uri = model_info["job"]["results"]["best_checkpoint_path_s3"]
        logger.info(
            f"Found best checkpoint path from {model_id=}: {checkpoint_path_uri}"
        )
    else:
        logger.info(f"Using {checkpoint_path_uri=} from manual input")

    # run evaluation
    logger.info("Running evaluation...")

    # generate a unique model id for this evaluation run
    eval_model_id = "EVALUATE_" + model_id + "_RUN_" + unique_id()

    with manager as ctx:
        return _evaluate(
            ctx=ctx,
            config=config,
            checkpoint_path_uri=checkpoint_path_uri,
            logger=logger,
            model_id=eval_model_id,
            name=eval_model_id,
        )


# see: scripts/launch_ctrl_dag.py
@attr.s(auto_attribs=True, kw_only=True)
class OSSEMGDAG(DagBase):
    config: dict[str, Any]
    train_machine_type: str
    eval_machine_type: str = attr.ib(default="HIMEM4")

    @classmethod
    def from_dict_config(cls, dict_config: DictConfig, **kwargs):
        config = OmegaConf.to_container(dict_config, resolve=True)
        return cls(config=config, **kwargs)

    @property
    def name(self):
        return "oss-emg-dag"

    @property
    def description(self):
        return "Open source EMG DAG for the Nature Platform Paper"

    def run_components(self, after: kfp_dsl.BaseOp):
        train_mtype = MachineType[self.train_machine_type]
        eval_mtype = MachineType[self.eval_machine_type]

        num_nodes = self.config.get("num_nodes", 1)

        def train_rank(node_rank: int):
            train_component = ctrl_comp(mtype=train_mtype)(train_component_fn)
            return train_component(
                config=self.config, node_rank=node_rank, num_nodes=num_nodes
            ).after(after)

        train_components = ctrl_dsl.Map(train_rank, list(range(num_nodes)))

        # run evaluation as a ctrl_comp
        eval_component = ctrl_comp(mtype=eval_mtype)(evaluate_after_train_fn)
        eval_component(config=self.config).after(train_components)

        return after


# see: scripts/launch_eval_only_dag.py
@attr.s(auto_attribs=True, kw_only=True)
class OSSEMGDagEvaluationOnly(DagBase):
    model_id: str = attr.ib(default="pytorch_9b15d8ee-b2cd-45e3-8313-f3fe0222c231")
    checkpoint_path_uri: str | None = attr.ib(default=None)
    eval_machine_type: str = attr.ib(default="HIMEM4")

    @property
    def name(self):
        return "oss-emg-dag-eval-only"

    @property
    def description(self):
        return "Open source EMG DAG for the Nature Platform Paper (eval only)"

    def run_components(self, after: kfp_dsl.BaseOp):
        eval_mtype = MachineType[self.eval_machine_type]

        # run evaluation from model id as a ctrl_comp
        evaluate_from_model_id_comp = ctrl_comp(mtype=eval_mtype)(
            evaluate_from_model_id
        )
        out = evaluate_from_model_id_comp(
            model_id=self.model_id,
            checkpoint_path_uri=self.checkpoint_path_uri,
        ).after(after)

        return out
