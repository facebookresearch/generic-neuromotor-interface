# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# TODO: delete before OSS

import click

from ctrl.logging import get_logger
from generic_neuromotor_interface.fb_train_dag import OSSEMGDagEvaluationOnly
from workflow.git_utils import Uncommitted

# example model id for HW
DEFAULT_MODEL_ID = "pytorch_9b15d8ee-b2cd-45e3-8313-f3fe0222c231"


@click.command()
@click.option(
    "--model-id",
    type=str,
    help=f"Model ID to use for evaluation. Default is {DEFAULT_MODEL_ID}",
    default=DEFAULT_MODEL_ID,
)
@click.option(
    "--checkpoint-path-uri",
    default=None,
    help="s3:// path to the model checkpoint to use; if not provided, the 'best' checkpoint will be used (best according to RP)",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Flag to run locally (default=False)",
)
@click.option(
    "--name",
    type=str,
    default="nature-eval-dag",
    help="Name of the DAG (default=nature-eval-dag)",
)
def main(
    model_id: str,
    checkpoint_path_uri: str | None = None,
    local: bool = False,
    name: str = "nature-eval-dag",
):

    logger = get_logger(__name__)
    dag = OSSEMGDagEvaluationOnly(
        model_id=model_id,
        checkpoint_path_uri=checkpoint_path_uri,
    )

    logger.info(f"Launching OSSEMGDagEvaluationOnly workflow with {model_id=}...")

    if local:
        logger.info("Launching locally...")
        dag.launch_local(name=name, disable_rp_reporting=True)
    else:
        logger.info("Launching to AWS...")
        dag.launch(name=name, uncommitted=Uncommitted.INCLUDE)


if __name__ == "__main__":
    main()
