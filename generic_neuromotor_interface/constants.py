# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

Task = Literal["discrete_gestures", "handwriting", "wrist"]

# TODO: Update these with the location of the open source data when released
S3_BUCKET = "fb-ctrl-general"
S3_PREFIX = "oss/generic-neuromotor-interface/030325-0"
S3_SUBSET_PREFIX = "oss/generic-neuromotor-interface/030325-0-small"

EMG_SAMPLE_RATE = 2000  # Hz
