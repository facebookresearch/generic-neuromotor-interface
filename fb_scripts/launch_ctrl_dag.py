# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# TODO: delete before OSS

import hydra

from ctrl.logging import get_logger
from generic_neuromotor_interface.fb_train_dag import OSSEMGDAG
from omegaconf import DictConfig
from workflow.git_utils import Uncommitted

# TODO: Move this script out of the open source repo
# see: from workflow.components.base import MachineType
TRAIN_MACHINE_TYPE = "G5"


@hydra.main(config_path="../config", config_name="wrist", version_base="1.1")
def main(config: DictConfig):

    logger = get_logger(__name__)
    dag = OSSEMGDAG.from_dict_config(
        dict_config=config,
        train_machine_type=TRAIN_MACHINE_TYPE,
    )

    version = "v3"
    experiment_name = f"{config.name}-test-{version}"

    logger.info("Launching OSSEMGDAG workflow...")

    if config.local:
        logger.info("Launching locally...")
        dag.launch_local(name=experiment_name, disable_rp_reporting=True)
    else:
        logger.info("Launching to AWS...")
        dag.launch(name=experiment_name, uncommitted=Uncommitted.INCLUDE)


if __name__ == "__main__":
    main()
