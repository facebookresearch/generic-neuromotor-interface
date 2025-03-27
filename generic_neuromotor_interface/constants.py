# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

Task = Literal["discrete_gestures", "handwriting", "wrist"]
EMG_SAMPLE_RATE = 2000  # Hz
