# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="generic_neuromotor_interface",
    version="0.1.0",
    description="Code for exploring surface electromyography (sEMG) data and "
    "training models associated with the paper `A generic noninvasive neuromotor "
    "interface for human-computer interaction`",
    author="CTRL-labs at Reality Labs, Meta",
    author_email="",
    packages=find_packages(),
    install_requires=[
        # Left empty so you use the conda environment.yml file
    ],
)
