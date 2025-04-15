# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Literal

Task = Literal["discrete_gestures", "handwriting", "wrist"]

S3_BUCKET = "fb-ctrl-oss"
S3_PREFIX = "neuromotor-data/emg_data.tar.gz"
S3_SUBSET_PREFIX = "neuromotor-data/emg_data_small.tar.gz"

EMG_SAMPLE_RATE = 2000  # Hz


class GestureType(enum.Enum):
    """Enumeration of discrete gesture types."""

    index_press = 0
    index_release = 1
    middle_press = 2
    middle_release = 3
    thumb_click = 4
    thumb_down = 5
    thumb_in = 6
    thumb_out = 7
    thumb_up = 8
