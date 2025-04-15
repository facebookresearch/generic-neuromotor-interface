# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any

import torch
from generic_neuromotor_interface.networks import (
    DiscreteGesturesArchitecture,
    MultivariatePowerFrequencyFeatures,
    RotationInvariantMPFMLP,
    WristArchitecture,
)

from hypothesis import given, strategies as st


class TestRotationInvariantMPFMLP(unittest.TestCase):

    @given(st.integers(min_value=1, max_value=16))
    def test_invalid_adjacent_cov(
        self,
        num_channels: int,
    ) -> None:
        max_adjacent_cov = num_channels // 2

        # test with max allowed adjacent cov
        RotationInvariantMPFMLP(
            num_channels=num_channels,
            num_freqs=1,
            hidden_dims=[1],
            num_adjacent_cov=max_adjacent_cov,
        )

        # test with one above
        with self.assertRaises(ValueError):
            RotationInvariantMPFMLP(
                num_channels=num_channels,
                num_freqs=1,
                hidden_dims=[1],
                num_adjacent_cov=max_adjacent_cov + 1,
            )


class TestRotationInvariantLSTM(unittest.TestCase):

    def test_forward(self) -> None:

        # set some constants
        batch_size = 5
        num_channels = 16
        num_tsteps = 500
        output_dim = 1

        # sample random data
        data = torch.randn(batch_size, num_channels, num_tsteps)

        # assemble network
        network = WristArchitecture(
            num_channels=num_channels,
            hidden_dims=[512],
            lstm_hidden_dim=512,
            lstm_num_layers=2,
            output_dim=output_dim,
        )

        # check that the number of parameters is exactly equal to how many
        # we had when running the scaling plots experiments
        num_params = sum(p.numel() for p in network.parameters())
        self.assertEqual(num_params, 4400129)

        # run network forward pass
        output = network(data)

        # check output shape
        output_num_tsteps = len(
            torch.arange(num_tsteps)[network.left_context :: network.stride]
        )
        self.assertEqual(output.shape, (batch_size, output_dim, output_num_tsteps))


DEFAULT_MPF_FREQUENCY_BINS = (
    (0, 50),
    (30, 100),
    (100, 225),
    (225, 375),
    (375, 700),
    (700, 1000),
)

WRIST_MODEL_MPF_PARAMS = {
    "window_length": 200,
    "stride": 40,
    "n_fft": 64,
    "fft_stride": 10,
}

HANDWRITING_MODEL_MPF_PARAMS = {
    "window_length": 160,
    "stride": 40,
    "n_fft": 64,
    "fft_stride": 10,
}


class TestMultivariatePowerFrequencyFeatures(unittest.TestCase):

    @given(
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=16),
        st.sampled_from([WRIST_MODEL_MPF_PARAMS, HANDWRITING_MODEL_MPF_PARAMS]),
    )
    def test_left_context(
        self,
        batch_size: int,
        num_channels: int,
        mpf_parameters: dict[str, Any],
    ) -> None:

        # assemble module
        module = MultivariatePowerFrequencyFeatures(**mpf_parameters)

        # check that module returns length 1 sequence when given
        # input of length left_context + 1
        data = torch.randn(batch_size, num_channels, module.left_context + 1)
        output = module(data)
        self.assertEqual(output.shape[-1], 1)

        # check that module raises RuntimeError when sequence length
        # is left_context
        with self.assertRaises(RuntimeError):
            data = torch.randn(batch_size, num_channels, module.left_context)
            module(data)


class TestDiscreteGesturesArchitecture(unittest.TestCase):

    def test_forward(self) -> None:

        # set some constants
        batch_size = 5
        num_channels = 16
        num_tsteps = 500

        # sample random data
        data = torch.randn(batch_size, num_channels, num_tsteps)

        # assemble network
        network = DiscreteGesturesArchitecture(input_channels=num_channels)

        # check that the number of parameters is exactly equal to how many
        # we had when running the scaling plots experiments
        num_params = sum(p.numel() for p in network.parameters())
        self.assertEqual(num_params, 6482953)

        # run network forward pass
        output = network(data)

        # check output shape
        output_num_tsteps = len(
            torch.arange(num_tsteps)[network.left_context :: network.stride]
        )
        self.assertEqual(output.shape, (batch_size, 9, output_num_tsteps))
