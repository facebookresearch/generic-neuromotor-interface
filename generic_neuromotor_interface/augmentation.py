# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RotationAugmentation:
    """Rotate EMG along the channel dimension by a random integer."""

    rotation: int = 1

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        rotation = np.random.choice(np.arange(-self.rotation, self.rotation + 1))
        return torch.roll(data, rotation, dims=-1)
