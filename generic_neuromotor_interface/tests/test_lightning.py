# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections.abc import Sequence

import torch
from generic_neuromotor_interface.lightning import WristModule
from generic_neuromotor_interface.networks import WristArchitecture
from hypothesis import given, strategies as st


class TestWristModule(unittest.TestCase):
    @staticmethod
    def build_module(
        num_channels: int,
        hidden_dims: Sequence[int],
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        output_dim: int,
    ) -> WristModule:
        return WristModule(
            network=WristArchitecture(
                num_channels=num_channels,
                hidden_dims=hidden_dims,
                lstm_hidden_dim=lstm_hidden_dim,
                lstm_num_layers=lstm_num_layers,
                output_dim=output_dim,
            ),
            optimizer=torch.optim.Adam,
        )

    @given(
        st.lists(
            elements=st.integers(min_value=1, max_value=5), min_size=1, max_size=5
        ),
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=3),
    )
    def test_forward(
        self,
        hidden_dims: Sequence[int],
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        output_dim: int,
    ) -> None:
        batch_size = 2
        num_channels = 16
        num_tsteps = 500

        # assemble lightning module
        module = self.build_module(
            num_channels=num_channels,
            hidden_dims=hidden_dims,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            output_dim=output_dim,
        )

        # sample random data
        data = torch.randn(batch_size, num_channels, num_tsteps)

        # run forward
        output = module(data)

        # check output shape
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], output_dim)

    @given(
        st.lists(
            elements=st.integers(min_value=1, max_value=5), min_size=1, max_size=5
        ),
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=1, max_value=3),
    )
    def test_step(
        self,
        hidden_dims: Sequence[int],
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        output_dim: int,
    ) -> None:
        batch_size = 2
        num_channels = 16
        num_tsteps = 500

        # assemble lightning module
        module = self.build_module(
            num_channels=num_channels,
            hidden_dims=hidden_dims,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            output_dim=output_dim,
        )

        # sample random data
        batch = {
            "emg": torch.randn(batch_size, num_channels, num_tsteps),
            "wrist_angles": torch.randn(batch_size, output_dim, num_tsteps),
        }

        # run step
        loss = module._step(batch)

        # check that loss value is positive (since it is an L1 loss)
        self.assertGreater(loss.item(), 0.0)
