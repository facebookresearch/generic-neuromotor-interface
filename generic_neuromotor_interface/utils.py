# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import torch
from torch import nn


def get_full_dataset_path(root: str, dataset: str) -> Path:
    """Add root prefix and .hdf5 suffix (if necessary) to dataset name."""
    path = Path(root).expanduser().joinpath(f"{dataset}")
    if not path.suffix:
        path = path.with_suffix(".hdf5")
    return path


def handwriting_collate(samples: list[dict[str, torch.Tensor]]):
    emg_batch = [sample["emg"] for sample in samples if sample]  # [(T, ...)]
    prompt_batch = [sample["prompts"] for sample in samples if sample]  # [(T)]

    # Batch of inputs and targets padded along time
    padded_emg_batch = nn.utils.rnn.pad_sequence(emg_batch)  # (T, N, ...)
    padded_prompt_batch = nn.utils.rnn.pad_sequence(prompt_batch)  # (T, N)

    # Lengths of unpadded input and target sequences for each batch entry
    emg_lengths = torch.as_tensor(
        [len(_input) for _input in emg_batch], dtype=torch.int32
    )
    prompt_lengths = torch.as_tensor(
        [len(target) for target in prompt_batch], dtype=torch.int32
    )

    return {
        "emg": padded_emg_batch.movedim(0, 2),  # (T, N, ...) -> # (N, T, ...)
        "prompts": padded_prompt_batch.T,  # (T, N) -> (N, T)
        "emg_lengths": emg_lengths,
        "prompt_lengths": prompt_lengths,
    }
