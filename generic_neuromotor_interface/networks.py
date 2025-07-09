# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import nn

LPadType = int | Literal["none", "steady", "full"]


class Permute(nn.Module):
    """Permute the dimensions of the input tensor.
    For example:
    ```
    Permute('NTC', 'NCT') == x.permute(0, 2, 1)
    ```
    """

    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        assert len(from_dims) == len(
            to_dims
        ), "Same number of from- and to- dimensions should be specified for"

        if len(from_dims) not in {3, 4, 5, 6}:
            raise ValueError(
                "Only 3, 4, 5, and 6D tensors supported in Permute for now"
            )

        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def get_inverse_permute(self) -> "Permute":
        "Get the permute operation to get us back to the original dim order"
        return Permute(from_dims=self.to_dims, to_dims=self.from_dims)

    def __repr__(self) -> str:
        return f"Permute({self.from_dims!r} => {self.to_dims!r})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class ReinhardCompression(nn.Module):
    """Dynamic range compression using the Reinhard operator."""

    def __init__(self, range: float, midpoint: float) -> None:
        super().__init__()
        self.range = range
        self.midpoint = midpoint

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.range * inputs / (self.midpoint + torch.abs(inputs))


class DiscreteGesturesArchitecture(nn.Module):
    """Discrete Gestures Network that combines ConvNet with stacked LSTM
    and projection layer.

    Parameters
    ----------
    input_channels : int
        Number of input channels for the ConvNet
    conv_output_channels : int
        Number of output channels from the ConvNet
    num_channels : int
        Number of intermediate channels in ConvNet layers
    num_conv_layers : int
        Number of convolutional layers in the ConvNet
    kernel_width : int
        Width of the convolutional kernels
    stride : int
        Stride of the convolutional layers
    lstm_hidden_size : int
        Number of features in the hidden state of the LSTM
    lstm_num_layers : int
        Number of stacked LSTM layers
    output_channels : int
        Number of gestures to predict
    """

    def __init__(
        self,
        input_channels: int = 16,
        conv_output_channels: int = 512,
        num_channels: int = 512,
        num_conv_layers: int = 1,
        kernel_width: int = 21,
        stride: int = 10,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 3,
        output_channels: int = 9,
    ) -> None:
        super().__init__()

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.left_context = kernel_width - 1
        self.stride = stride

        # Reinhard Compression
        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)

        # Conv1d layer
        self.conv_layer = nn.Conv1d(
            input_channels,
            conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
        )

        # Relu
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

        # Layer normalization
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1,
        )

        # Layer normalization
        self.post_lstm_layer_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)

        # Feedforward projection layer
        self.projection = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, sequence_length)

        Returns
        -------
        output : torch.Tensor
            - Output tensor of shape (batch_size, num_gestures, sequence_length)
        """

        # Reinhard Compression
        x = self.compression(inputs)

        # Conv1d layer
        x = self.conv_layer(x)

        # Relu
        x = self.relu(x)

        # Dropout
        x = self.dropout(x)

        # Layer normalization
        # (batch_size, conv_output_channels, sequence_length)
        # -> (batch_size, sequence_length, conv_output_channels)
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)

        # Stacked LSTM layers
        x, _ = self.lstm(x)

        # Layer normalization
        x = self.post_lstm_layer_norm(x)

        # Feedforward projection layer
        x = self.projection(x)
        x = x.permute(0, 2, 1)

        return x


class WristArchitecture(nn.Module):
    def __init__(
        self,
        num_channels: int,
        hidden_dims: Sequence[int],
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        output_dim: int,
        offsets: Sequence[int] | Sequence[float] = (-1, 0, 1),
        num_adjacent_cov: int = 3,
    ) -> None:
        """
        [RotationInvariantMPFMLP] -> [LSTM] -> [LeakyReLU] -> [Linear]

        Takes as input batches of shape
            (batch_size, frequency, channels, channels, time)
        and produces outputs of shape
            (batch_size, output_dim, time)

        Parameters
        ----------
        num_channels : int
            passed to RotationInvariantMPFMLP
        hidden_dims : Sequence[int]
            passed to RotationInvariantMPFMLP
        lstm_hidden_dim : int
            hidden dim for LSTM state
        lstm_num_layers : int
            number of LSTM layers
        output_dim : int
            output dim of network
        offsets : Union[Sequence[int], Sequence[float]], optional
            passed to RotationInvariantMPFMLP, by default (-1, 0, 1)
        num_adjacent_cov : int | None, optional
            passed to RotationInvariantMPFMLP, by default 3
        """

        super().__init__()

        # Convert raw EMG to MPF features
        mpf_frequency_bins = (
            (0, 50),
            (30, 100),
            (100, 225),
            (225, 375),
            (375, 700),
            (700, 1000),
        )
        self.mpf_featurizer = MultivariatePowerFrequencyFeatures(
            window_length=200,
            stride=40,
            n_fft=64,
            fft_stride=10,
            frequency_bins=mpf_frequency_bins,
        )

        # Rotation-invariance module
        self.rotation_invariant_mlp = RotationInvariantMPFMLP(
            num_channels=num_channels,
            num_freqs=len(mpf_frequency_bins),
            hidden_dims=hidden_dims,
            offsets=offsets,
            num_adjacent_cov=num_adjacent_cov,
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dims[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Readout layers
        self.relu = nn.LeakyReLU()
        self.linear_out = nn.Linear(lstm_hidden_dim, output_dim)

        # Left context size and stride are given by
        # that of MPF features
        self.left_context = self.mpf_featurizer.left_context
        self.stride = self.mpf_featurizer.stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.mpf_featurizer(inputs)
        x = self.rotation_invariant_mlp(x)
        x = x.permute(0, 2, 1)  # (batch, channels, time) -> (batch, time, channels)
        # NOTE: We intentionally don't propagate from any previous forward calls,
        # nor do we propagate the hidden state to future forward calls.
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = x.permute(0, 2, 1)  # (batch, time, output_dim) -> (batch, output_dim, time)
        return x


class RotationInvariantMPFMLP(nn.Module):
    """Mean-pools over same projection of channel-rotated MPF features

    Rotates Multivariate Power Frequency (MPF) features by discrete
    electrode offsets, vectorizes, and mean-pools over projections of
    the vectorized inputs through the same MLP.

    Vectorization includes subselecting a given number of off-diagonals
    to keep only covariances between adjacent channels, assuming circular
    adjacency.

    Takes as input batches of MPF features of shape
        (batch_size, frequency, channels, channels, time)
    and produces outputs of shape
        (batch_size, output_dim, time)

    Parameters
    ----------
    num_channels : int
        Number of channels in the MPF covariances, i.e. each MPF covariance
        matrix should be size (num_channels, num_channels)
    num_freqs : int
        Number of frequencies in the MPF features
    hidden_dims : Sequence[int]
        Dimensions of MLP hidden layers.
    offsets : Union[Sequence[int], Sequence[float]], optional
        Integer band rotations to apply, by default (-1, 0, 1)
    num_adjacent_cov : int | None, optional
        Keep covariance values from this many off-diagonals, by default 3.
    """

    def __init__(
        self,
        num_channels: int,
        num_freqs: int,
        hidden_dims: Sequence[int],
        offsets: Sequence[int] | Sequence[float] = (-1, 0, 1),
        num_adjacent_cov: int = 3,
    ) -> None:

        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty")
        self.offsets = offsets
        self.activation = nn.LeakyReLU()
        self.vectorize = VectorizeSymmetricMatrix(
            num_channels=num_channels,
            num_adjacent_cov=num_adjacent_cov,
        )
        self.flatten = torch.nn.Flatten(start_dim=3)
        self.fully_connected_layers = nn.ModuleList()
        dim = num_freqs * min(
            num_channels * (num_adjacent_cov + 1),
            num_channels * (num_channels + 1) // 2,
        )
        for hidden_dim in hidden_dims:
            self.fully_connected_layers.append(nn.Linear(dim, hidden_dim))
            dim = hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # Rotate inputs by each offset
        x = torch.stack(
            [
                inputs.roll(
                    shifts=[offset, offset],
                    dims=[2, 3],  # shift channel dimensions
                )
                for offset in self.offsets
            ],
            dim=-1,
        )

        # Permute from (batch, frequency, channel, channel, time, rotations)
        # to (batch, time, rotations, frequency, channel, channel)
        x = x.permute(0, 4, 5, 1, 2, 3)

        # Vectorize and subselect off-diagonals
        x = self.vectorize(x)

        # Flatten across frequency and covariance dimensions
        x = self.flatten(x)

        # Pass each vectorized, flattened, and rotated covariance
        # the same the same MLP
        for layer in self.fully_connected_layers:
            x = self.activation(layer(x))

        # Average outputs over rotations
        x = torch.mean(x, dim=2, keepdim=False)

        # Permute from (batch, time, output_dim) back
        # to (batch, output_dim, time)
        x = x.permute(0, 2, 1)

        return x


class VectorizeSymmetricMatrix(nn.Module):
    """Vectorize a symmetric matrix by concatenating diagonals in upper triangle.

    Assumes inputs with 6 dimensions, the last two of which have identical size,
    corresponding to the symmetric matrix of shape (num_channels, num_channels).

    Subselects a given number of off-diagonals to keep only covariances between
    adjacent channels, assuming circular adjacency.

    Parameters
    ----------
    num_channels : int, optional
        Number of channels, by default 16
    num_adjacent_cov : int | None, optional
        Keep covariance values of this many adjacent channel pairs, by default
        None, in which case all (upper triangular) covariances are kept.
    """

    def __init__(
        self,
        num_channels: int,
        num_adjacent_cov: int | None = None,
    ) -> None:
        super().__init__()

        max_adjacent_cov = num_channels // 2

        if num_adjacent_cov is None:
            num_adjacent_cov = max_adjacent_cov

        if num_adjacent_cov > max_adjacent_cov:
            raise ValueError(
                f"invalid value num_adjacent_cov={num_adjacent_cov=}, "
                f"which must be less than {num_channels // 2=}"
            )

        # get indices of upper triangular values
        # since lower triangular values are redundant
        triu_row_indices, triu_col_indices = torch.triu_indices(
            num_channels, num_channels
        )

        # get indices of given adjacent channel covariances
        adjacent_cov_mask = self._get_adjacent_cov_mask(num_channels, num_adjacent_cov)
        adjacent_cov_triu_indices_mask = (
            adjacent_cov_mask[triu_row_indices, triu_col_indices] == 1
        )

        # 2D index to 1D and pre-apply mask
        row_indices = triu_row_indices[adjacent_cov_triu_indices_mask]
        col_indices = triu_col_indices[adjacent_cov_triu_indices_mask]
        self.flattened_matrix_indices = num_channels * row_indices + col_indices

    @staticmethod
    def _get_adjacent_cov_mask(num_channels: int, num_adjacent_diagonals: int):
        """Build a mask for the adjacent diagonals."""
        mask = torch.eye(num_channels)
        for i in range(num_channels):
            for j in range(num_adjacent_diagonals):
                mask[i, (i + j + 1) % num_channels] = 1
                mask[i, (i - j - 1) % num_channels] = 1
        return mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.flatten(4)
        vec = inputs[:, :, :, :, self.flattened_matrix_indices]
        return vec


class MultivariatePowerFrequencyFeatures(nn.Module):
    """Convert raw EMG to multivariate power frequency (MPF) features.

    Takes as input batches of raw EMG of shape
        (batch_size, channels, time)
    and produces outputs of shape
        (batch_size, frequency, channels, channels, time)

    Parameters
    ----------
    window_length : int
        Window length for computation of MPF features. Must be larger than `n_fft`.
    stride : int
        Number of samples to stride between consecutive MPF windows
    n_fft : int
        Number of FFT samples (see `n_fft` parameter of `torch.stft`). Also
        determines the size of the STFT windows.
    fft_stride : int
        Stride of windows used to compute the STFT. Must be a multiple of `n_fft`.
    fs : float
        Sampling frequency of the input EMG, by default 2000.0.
    frequency_bins : list[tuple[float, float]] or None, optional
        Average over FFT frequencies within each bin. For example, to produce two
        bins, one between 100 and 200Hz, and the other between 300 and 500Hz,
        set `frequency_bins` to [(100, 200), (300, 500)]. By default None, in which
        case all FFT frequencies are returned as is.
    """

    def __init__(
        self,
        window_length: int,
        stride: int,
        n_fft: int,
        fft_stride: int,
        fs: float = 2000.0,
        frequency_bins: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        if window_length < n_fft:
            raise ValueError("window_length must be greater than n_fft")
        if fft_stride > n_fft:
            raise ValueError("fft_stride must be lower than n_fft.")
        if fft_stride > stride:
            raise ValueError("stride must be greater than fft_stride")
        if stride % fft_stride != 0:
            raise ValueError("stride must be a multiple of fft_stride")

        self.window_length = window_length
        self.stride = stride
        self.n_fft = n_fft
        self.fft_stride = fft_stride
        self.fs = fs
        self.frequency_bins = frequency_bins

        # construct Hanning window for STFT
        window = torch.hann_window(self.n_fft, periodic=False)
        self.register_buffer("window", window)
        self.window_normalization_factor = torch.linalg.vector_norm(self.window)

        # if frequency bins specified, construct mask for averaging over frequencies
        # in each bin
        if self.frequency_bins is not None:
            freq_masks = self._build_freq_masks(
                self.n_fft, self.fs, self.frequency_bins
            )
            self.register_buffer("freq_masks", freq_masks)

        # calculate left context size
        self.left_context = self.window_length - self.fft_stride + self.n_fft - 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, _ = inputs.shape
        num_freqs = self.n_fft // 2 + 1

        # calculate spectrogram
        outputs = (
            torch.stft(
                inputs.reshape(
                    batch_size * num_channels,
                    -1,  # flatten over batch and channel dims,
                ),
                n_fft=self.n_fft,
                hop_length=self.fft_stride,
                window=self.window,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            / self.window_normalization_factor
        )  # (batch_size * channels, freqs, time)

        # reshape into strided windows
        outputs = outputs.unfold(
            dimension=-1,
            size=self.window_length // self.fft_stride,
            step=self.stride // self.fft_stride,
        )  # (batch_size * channels, freqs, windows, window_size)

        # reshape to get back channel dimension
        _, _, num_windows, window_size = outputs.shape
        outputs = outputs.reshape(
            batch_size,
            num_channels,
            num_freqs,
            num_windows,
            window_size,
        )

        # put channel dim in second-to-last dimension for
        # cross-spectral density calculation
        outputs = outputs.transpose(
            1, 3
        )  # (batch_size, windows, freqs, channels, window_size)

        # calculate cross-spectral density
        outputs = self._compute_strided_cross_spectral_density(outputs)

        # average over frequencies in each frequency band
        if self.frequency_bins is not None:
            outputs = torch.stack(
                [
                    (outputs * freq_mask).sum(2) / freq_mask.sum(2)
                    for freq_mask in self.freq_masks.unbind(2)
                ],
                dim=2,
            )

        # calculate matrix logarithm (Barachant et al. 2012)
        eigvals, eigvecs = torch.linalg.eigh(outputs)
        eigvals = eigvals.log().nan_to_num(nan=0.0, neginf=0.0)
        outputs = (eigvecs * eigvals.unsqueeze(dim=-2)) @ eigvecs.transpose(-1, -2)

        # reshape to (batch_size, freq, channels, channels, time)
        outputs = outputs.permute(0, 2, 3, 4, 1)

        return outputs

    @staticmethod
    def _build_freq_masks(
        n_fft: int,
        fs: float,
        frequency_bins: Sequence[tuple[float, float]],
    ) -> torch.Tensor:

        # get STFT frequencies
        freqs_hz = torch.fft.fftfreq(n_fft, d=1.0 / fs)[: (n_fft // 2 + 1)].abs()

        # construct masks for each bin
        freq_masks = torch.stack(
            [
                torch.logical_and(freqs_hz > start_freq, freqs_hz <= end_freq)
                for start_freq, end_freq in frequency_bins
            ]
        ).to(dtype=torch.uint8)

        # Unsqueeze to (batch_size, time, freq_bins, freqs, channels, channels)
        # format to broadcast over the frequency dim of the MPF features of shape
        # (batch_size, time, freqs, channels, channels)
        return freq_masks.reshape(1, 1, len(frequency_bins), len(freqs_hz), 1, 1)

    @staticmethod
    def _compute_strided_cross_spectral_density(inputs: torch.Tensor) -> torch.Tensor:
        """Compute cross-spectral density of strided windows of spectrogram features.

        Assumes inputs of shape
            (..., channels, time)

        Cross-spectral density between each pair of channels is computed, resulting
        in an output of shape
            (..., channels, channels)
        """

        input_dims = inputs.shape
        num_channels, window_size = input_dims[-2:]

        # Flatten over batch, frequency, and window dimensions to process all at once
        outputs = inputs.reshape(
            -1, num_channels, window_size
        )  # (batch_size * windows * freqs, channels, window_size)

        # Calculate cross-spectral density:
        outputs = (outputs @ outputs.transpose(-2, -1).conj()) / window_size
        outputs = outputs.abs().pow(2)

        # Reshape to restore the original batch, frequency, and window dimensions
        outputs = outputs.reshape(*input_dims[:-2], num_channels, num_channels)

        return outputs

    def compute_time_downsampling(self, emg_lengths: torch.Tensor) -> Sequence[int]:
        cospectrum_len = 1 + (emg_lengths - self.n_fft) // self.fft_stride
        return (cospectrum_len - self.window_length // self.fft_stride) // (
            self.stride // self.fft_stride
        ) + 1


class _AxesMask(nn.Module):
    """Samples and applies a mask along a given axis of a
    `MultivariatePowerFrequencyFeatures`.

    Takes as input batches of Multivariate Power Frequency Feature batch of shape
        (batch_size, frequency, channels, channels, time)
    and produces outputs of the same shape, where a mask of length `max_mask_length`
    is applied to each of the specified axes (e.g., axes=[1,4] -> mask applied to
    frequency and time).

    The filler value of the mask is specified by `mask_value`.

    Parameters
    ----------
    max_mask_length:
        Max length of each mask.
    axes:
        Axes to apply the masks over. When this contains multiple axes, they should all
        be of the same length. The same mask then will be applied to each of the axis
        specified. This can be used, for example, to apply channel masking on
        convariance computed across channels.
    mask_value:
        Mask value to apply.
    """

    def __init__(
        self,
        max_mask_length: int,
        axes: tuple[int, ...],
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()

        for axis in axes:
            assert axis > 0, "Cannot mask batch dim"

        self.max_mask_length = max_mask_length
        self.axes = axes
        self.mask_value = mask_value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs

        N = inputs.size(0)
        device = inputs.device
        dtype = inputs.dtype
        data_length = inputs.size(self.axes[0])
        max_mask_length = min(self.max_mask_length, data_length)
        value = torch.rand(N, device=device, dtype=dtype) * max_mask_length
        min_value = torch.rand(N, device=device, dtype=dtype) * (data_length - value)
        mask_start = min_value.long()
        mask_end = mask_start + value.long()

        for _ in range(inputs.ndim - 1):
            mask_start = mask_start.unsqueeze(-1)
            mask_end = mask_end.unsqueeze(-1)

        # Apply the same masks along the specified axes.
        idx = torch.arange(0, data_length, device=device, dtype=dtype)
        mask_idx = (idx >= mask_start) & (idx < mask_end)
        x = inputs
        for axis in self.axes:
            assert (
                x.size(axis) == data_length
            ), "All axes to mask must have the same length"
            x = x.transpose(axis, -1)
            x = x.masked_fill(mask_idx, self.mask_value)
            x = x.transpose(-1, axis)

        return x


class MaskAug(nn.Module):
    """Implementation of SpecAugment, which applies time- and frequency-aligned masks
    to these spectral features to a `MultivariatePowerFrequencyFeatures`.

    Takes as input batches of Multivariate Power Frequency Feature batch of shape
        (batch_size, frequency, channels, channels, time)
    and produces outputs of the same shape, where a for each dimensions in `dims`,
    we applied `max_num_masks` mask of length `max_mask_length` is applied to each
    of the specified axes(e.g., axes=[1,4] -> mask applied to frequency and time).

    The filler value of the mask is specified by `mask_value`.

    Parameters
    ----------
    max_num_masks:
        Max number of masks corresponding to each axis.
    max_mask_lengths:
        Max length of masks corresponding to each axis.
    dims:
        Ordered coordinates to mask, by default "TF".
    axes_by_coord:
        Mapping of supported axes to their indices. Defaults to
        {'N':[0], 'F':[1], 'C':[2, 3], 'T':[4]}.
    mask_value:
        Mask value to apply.
    """

    def __init__(
        self,
        max_num_masks: list[int],
        max_mask_lengths: list[int],
        dims: str = "TF",
        axes_by_coord: dict[str, tuple[int, ...]] | None = None,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.attrs = {
            "max_num_masks": max_num_masks,
            "max_mask_lengths": max_mask_lengths,
            "dims": dims,
            "axes_by_coord": axes_by_coord,
            "mask_value": mask_value,
        }
        assert len(max_num_masks) == len(dims)
        assert len(max_mask_lengths) == len(dims)

        if axes_by_coord is None:
            axes_by_coord = {"N": (0,), "F": (1,), "C": (2, 3), "T": (4,)}

        self.max_num_masks = []
        self.masks = nn.ModuleList()
        for dim in "CFT":
            if dim not in dims or dim not in axes_by_coord.keys():
                continue

            dim_idx = dims.index(dim)
            self.max_num_masks.append(max_num_masks[dim_idx])
            self.masks.append(
                _AxesMask(
                    max_mask_length=max_mask_lengths[dim_idx],
                    axes=axes_by_coord[dim],
                    mask_value=mask_value,
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs

        # Apply masks
        x = inputs
        for i, mask in enumerate(self.masks):
            n_masks = torch.randint(self.max_num_masks[i] + 1, size=())
            for _ in range(int(n_masks.item())):
                x = mask(x)

        return x


def TimeReductionLayer(
    stride: int,
    lpad: LPadType = 0,
) -> nn.Module:
    """Time reduction (or temporal frame stacking).

    Takes as input batches of shape
        (batch_size, T, C)
    and produces outputs of shape
        (batch_size, T', C')
    where T' = (time - stride + 1)) /+ stride and C'=C*stride
    (using `/+` to denote a "round up" integer division instead of the
    default truncated `//` integer division)
    """
    return SlicedSequential(
        Window(
            kernel_size=stride,
            stride=stride,
            lpad=lpad,
        ),
        Permute("NTCS", "NTSC"),
        nn.Flatten(start_dim=2),
    )


class Window(nn.Module):
    """Extracts sliding windows from a larger input tensor.

    Takes as input batches of shape
        (N, T, ...)
    and produces outputs of shape
        (N, T', ..., kernel_size)

    Given an [N, T, ...] input tensor we extract contiguous (potentially
    overlapping) windows of samples into a [N, T', ..., kernel_size]
    tensor, where T' = (T - dilation * (kernel_size - 1)) /+ stride
    (using `/+` to denote a "round up" integer division instead of the
    default truncated `//` integer division).

    Parameters
    ----------
    kernel_size :
        Size of the windows we want to extract
    stride :
        Stride between consecutive windows of data.
    dilation :
        How to stride samples *within* a window of data.
    lpad :
        Number of zero-padding samples on the left.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        lpad: LPadType = 0,
    ) -> None:
        super().__init__()

        self.receptive_field = 1 + dilation * (kernel_size - 1)
        self.stride = stride
        self.state_size = self.receptive_field - self.stride

        if self.receptive_field < self.stride:
            raise ValueError(
                f"{self.receptive_field=} < {self.stride=} is not supported!"
            )

        if isinstance(lpad, int):
            self.lpad: int = lpad
        else:
            if lpad not in {"none", "steady", "full"}:
                raise ValueError(f"Invalid {lpad=}")
            self.lpad = {
                "none": 0,
                "steady": self.state_size,
                "full": self.receptive_field - 1,
            }[lpad]

        if self.lpad >= self.receptive_field:
            raise ValueError(
                f"{self.lpad=} should be less than {self.receptive_field=}"
            )

        self.extra_left_context = self.receptive_field - 1 - self.lpad

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        if self.state_size > 0:
            # Pad the input
            buffered = F.pad(inputs, (0, 0, self.state_size, 0), "constant", 0)
        else:
            buffered = inputs

        windows = buffered.unfold(
            dimension=1, size=self.receptive_field, step=self.stride
        )
        return windows[..., :: self.dilation]


class Residual(nn.Module):
    """Container to handle residual/skip connection.

    This module can handle residual connections where the child module
    can have a different input and output temporal shapes. E.g., in the
    conformer model, the input and output shapes are different due to the
    temporal convolution kernel size and stride. If the input and output
    are different, then the module will slice the input tensor to match the
    output tensor.

    Parameters
    ----------
    child :
        `nn.Module` to wrap within.
    dropout :
        Residual dropout probability.
    weight :
        Weight of the residual connection, applied as
        outputs = child(inputs) + weight * dropout(inputs)
    """

    def __init__(
        self, child: nn.Module, dropout: float = 0.0, weight: float = 1.0
    ) -> None:
        super().__init__()

        self.child = child
        self.extra_left_context = self.child.extra_left_context
        self.stride = self.child.stride

        self.dropout = nn.Dropout(dropout)
        self.weight = weight

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        residual = inputs[:, self.extra_left_context :: self.stride]

        x = self.child(inputs)
        x = x + self.weight * self.dropout(residual)
        return x


class SlicedSequential(nn.Sequential):
    """Wrapper on top of of nn.Sequential that tracks the extra_left_context and stride
    of the linear chain of modules.

    The module automatically computes a `slice` object alongside the model forward
    pass prediction that can be used to slice the input tensor to match the temporal
    dimension of the output tensor. This is used in the handwriting task to compute
    the correct emission length.

    At initialization, the module will go through the list of modules and check if they
    have the extra_left_context and stride attributes. If they do, it will update the
    total extra_left_context and stride of the SlicedSequential module sequence.

    Important: It is assumed that all modules in the sequence that will have an impact
    on the time dimention have the extra_left_context and stride attributes.

    Parameters
    ----------
    modules :
        sequence of Ordered dict of `nn.Module` to wrap within.
    """

    def __init__(self, *modules) -> None:
        super().__init__(*modules)
        self.extra_left_context, self.stride = self.__get_extra_left_context_and_stride(
            list(self)
        )

    @staticmethod
    def __get_extra_left_context_and_stride(seq) -> tuple[int, int]:
        left, stride = 0, 1
        for mod in seq:
            if hasattr(mod, "extra_left_context") and hasattr(mod, "stride"):
                left += mod.extra_left_context * stride
                stride *= mod.stride
        return left, stride


class MultiHeadAttention(nn.Module):
    r"""Causal MultiheadAttention implementation.

    Causality is imposed through windowing of the input data, i.e., a sequence.
        See full description of how we achieve this in the
        `_init_and_register_attn_mask` method.


    Takes as input batches of shape
        (N, T_in, input_dim)
    Input tensor if split into windowed chunks of shape (to ensure causality)
        (N, T_out, window_size, input_dim)
    and produces outputs of shape after attention
        (N, T_out, input_dim)

    Parameters
    ----------
    input_dim: .
    num_heads: Number of parallel attention heads. Note that ``embed_dim``
        will be split across ``num_heads`` (i.e. each head will have dimension
        ``embed_dim // num_heads``).
    window_size: Receptive field of the attention.
    stride: Stride used to compute the windows (receptive field) over which
        the attention is computed.
    lpad: Left padding when creating the attention windows. Defaults to "steady"
        which adds zero padding to the left of the sample such that the output
        time length is not affected by the receptive field size.
    dropout: Dropout probability on ``attn_output_weights``.
        Default: ``0.0`` (no dropout).
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        window_size: int,
        stride: int = 1,
        lpad: LPadType = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.op = self._get_attn_op(
            dropout=dropout,
        )

        self.window = Window(
            kernel_size=window_size,
            stride=stride,
            lpad=lpad,
        )

        self.lpad = self.window.lpad
        if self.lpad > 0 and stride > 1:
            raise NotImplementedError(
                "MultiHeadAttention currently only supports unit stride when lpad > 0"
            )

        self.extra_left_context = self.window.extra_left_context
        self.stride = self.window.stride
        self._init_and_register_attn_mask()

    def _register_attn_mask(self, attn_mask: torch.Tensor) -> None:
        self.register_buffer("attn_mask", attn_mask)

    def _attn_params(
        self,
        windows: torch.Tensor,  # (N, T_out, window_size, input_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, kv, _ = self._attn_params_op(windows)
        return q, kv, kv

    def _get_attn_op(
        self,
        dropout: float,
    ) -> torch.nn.Module:
        return nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _init_and_register_attn_mask(self) -> None:
        # Initialize attention mask of shape (window_size, window_size) to mask
        # out the left padding of zeros corresponding to `lpad`.
        #
        # For window_size = 4, this would be:
        #   1 1 1 0
        #   1 1 0 0
        #   1 0 0 0
        #   0 0 0 0
        #
        # Note that `torch.nn.MultiHeadAttention` convention is 0 if the corresponding
        # position is allowed to attend.
        attn_mask = torch.ones(self.window_size, self.window_size, dtype=torch.bool)
        attn_mask = attn_mask.triu(diagonal=1).flip(dims=[1])
        # Repeat the attn mask for causal sliding windows along the temporal axis to
        # reflect the manner in which each modality stream are concatenated, such that
        # attention can be applied to the causal histories of all streams in one
        # attentional receptive field.
        self._register_attn_mask(attn_mask)

    def _attn_params_op(
        self,
        windows: torch.Tensor,  # (N, T_out, window_size, input_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # kv: (N * T_out, window_size, input_dim)
        windows = windows.reshape(-1, self.window_size, self.input_dim)

        # Query comprises of the last sample of each of the sliding windows.
        # query: (N * T_out, input_dim)
        query = windows[:, -1]

        kv = windows.transpose(1, 2).reshape(-1, self.window_size, self.input_dim)
        return query, kv, kv

    def _attn_mask_op(
        self,
        windows: torch.Tensor,  # (N, T_out, window_size, input_dim)
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        T_out = windows.shape[1]

        # Generate attention mask:
        #
        # 1. Grab attention mask for output samples that potentially correspond to
        #    warmup (i.e., attention_window < receptive_field) by clamping the range
        #    [t_start, t_end) to [extra_left_context, receptive_field - 1).
        warmup_idx = slice(
            max(t_start, self.extra_left_context),
            min(t_end, self.window_size - 1),
            self.stride,
        )
        attn_mask_warmup = self.attn_mask[warmup_idx]

        # 2. Remaining samples in T_out correspond to the post-warmup steady state
        #    (i.e., attention_window == window_size).
        T_out_warmup = len(attn_mask_warmup)
        T_out_steady = T_out - T_out_warmup
        assert T_out_steady >= 0

        # 3. T_out_steady samples have full receptive field of valid samples and thus
        #    no masking is needed. Replicate the final row of self.attn_mask
        #    (essentially a tensor of zeros) T_out_steady times to achieve this.
        #    We use torch.expand here which simply creates a view unlike torch.repeat
        #    to avoid memory allocation and copy.
        # attn_mask_steady: (T_out_steady, window_size)
        attn_mask_steady = self.attn_mask[-1:, :].expand(T_out_steady, -1)

        # 4. Create the full mask for T_out by concatenating the corresponding masks
        #    for warmup and steady state samples.
        # attn_mask: (T_out, window_size)
        attn_mask = torch.cat([attn_mask_warmup, attn_mask_steady], dim=0)
        return attn_mask

    def _attn_mask(
        self,
        windows: torch.Tensor,  # (N, T_out, window_size, input_dim)
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        N, T_out = windows.shape[:2]
        attn_mask = self._attn_mask_op(windows, t_start, t_end)
        # Reshape and expand attn_mask from (T_out, window_size) to
        # (N * T_out * num_heads, window_size) to
        # match query/key/value.  In `torch.nn.MultiHeadAttention` parlance, L=1,
        # S=window_size.
        attn_mask = (
            attn_mask.reshape(1, T_out, 1, -1)
            .expand(N, T_out, self.op.num_heads, -1)
            .reshape(N * T_out * self.op.num_heads, -1)
        )
        return attn_mask

    def _windows(self, inputs: torch.Tensor) -> torch.Tensor:
        N, T_in = inputs.shape[:2]
        # Window buffer can not have tensor structure for time series.
        inputs = inputs.reshape((N, T_in, self.input_dim))
        windows = self.window(inputs)
        # Reshape windows from (N, T_out, input_dim, window_size) to
        # (N, T_out, window_size, input_dim)
        windows = windows.transpose(-1, -2).reshape(
            (N, -1, self.window_size, self.input_dim)
        )
        return windows

    def _forward(
        self,
        inputs: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        windows: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a QKV attention operation with optional attention masks.

        Parameters
        ----------
        inputs:
            Input tensor of shape [N, T_in, ...]
        query:
            Query embeddings of shape [N, q_len, embed_dim]
        key:
            Key embeddings of shape [N, k_len, embed_dim]
        value:
            Value embeddings of shape [N, k_len, embed_dim]
        attn_mask:
            Attention mask of shape [N * num_heads, q_len, k_len]
        windows:
            Input tensor of shape [N, T_out, window_length, input_dim]
        """
        N, T_in = inputs.shape[:2]

        attn_mask = self._attn_mask(windows, 0, T_in)

        output, _ = self.op(
            query=query.unsqueeze(1),
            key=key,
            value=value,
            attn_mask=attn_mask.unsqueeze(1),
        )
        output = output.reshape(N, -1, self.input_dim).unsqueeze(
            2
        )  # Unmerge T_out from batch dim

        return output

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        windows = self._windows(inputs)
        query, key, value = self._attn_params(windows)

        return self._forward(inputs, query, key, value, windows)


class Conv1d(nn.Module):
    """Applies 1d convolution, where 1 dimension is across time

    Input is expected to be in NTC format, where
        NTC = Batch x Time x Channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.permute_forward = Permute("NTC", "NCT")
        self.permute_back = self.permute_forward.get_inverse_permute()

        self.receptive_field = 1 + dilation * (kernel_size - 1)
        self.state_size = self.receptive_field - stride

        self.stride = stride
        self.extra_left_context = self.receptive_field - 1

        self.net = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        inputs = self.permute_forward(inputs)
        outputs = self.net(inputs)
        outputs = self.permute_back(outputs)

        return outputs


def ConformerEncoderBlock(
    input_dim: int,
    ffn_dim: int,
    kernel_size: int,
    stride: int,
    num_heads: int,
    attn_window_size: int,
    attn_lpad: LPadType = "steady",
    dropout: float = 0.0,
) -> nn.Module:
    """Builder function for a single conformer encoder block.

    (ff_block1): FF block
    (attn_block): Attention block
    (conv_block): Convolution block
    (ff_block2): FF block
    (layer_norm)
    """
    ff_block1: nn.Module = Residual(
        SlicedSequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout),
        ),
        weight=0.5,
    )
    ff_block2: nn.Module = Residual(
        SlicedSequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout),
        ),
        weight=0.5,
    )

    if attn_window_size > 0:
        attn_block: nn.Module = Residual(
            SlicedSequential(
                nn.LayerNorm(input_dim),
                nn.Unflatten(dim=-1, unflattened_size=(1, input_dim)),
                MultiHeadAttention(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    window_size=attn_window_size,
                    stride=1,
                    lpad=attn_lpad,
                    dropout=dropout,
                ),
                nn.Flatten(start_dim=-2),
                nn.Dropout(dropout),
            )
        )
    else:
        attn_block = nn.Identity()

    if kernel_size > 0:
        conv_block: nn.Module = Residual(
            SlicedSequential(
                nn.LayerNorm(input_dim),  # NTC
                nn.Linear(input_dim, 2 * input_dim),  # pointwise conv
                nn.GLU(dim=-1),
                Conv1d(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=input_dim,
                ),
                nn.LayerNorm(input_dim),
                nn.SiLU(),
                nn.Linear(input_dim, input_dim),  # pointwise conv
                nn.Dropout(dropout),
            )
        )
    else:
        conv_block = nn.Identity()

    return SlicedSequential(
        OrderedDict(
            {
                "ff_block1": ff_block1,
                "attn_block": attn_block,
                "conv_block": conv_block,
                "ff_block2": ff_block2,
                "layer_norm": nn.LayerNorm(input_dim),
            }
        )
    )


def ConformerEncoder(
    input_dim: int,
    ffn_dim: int,
    kernel_size: int | list[int],
    stride: int | list[int],
    num_heads: int,
    attn_window_size: int | list[int],
    num_layers: int | None,
    conv_lpad: LPadType | list[LPadType] = 0,
    attn_lpad: LPadType | list[LPadType] = "steady",
    dropout: float = 0.0,
) -> nn.Module:
    """Builder function for the multi-layered conformer encoder.

    Any attribute that is specified as a value and not a list of length == num_layer,
    the value will be set to be constant for all the layers.
    """
    assert num_layers is not None, "num_layers cannot be inferred, must be specified"

    # Convert int specifications to a list of ints, one value per block
    if not isinstance(attn_window_size, list) and not isinstance(
        attn_window_size, ListConfig
    ):
        # Constant attn window size for all blocks
        attn_window_size = [attn_window_size] * num_layers
    if not isinstance(kernel_size, list) and not isinstance(kernel_size, ListConfig):
        # Constant conv kernel size all blocks
        kernel_size = [kernel_size] * num_layers
    if not isinstance(stride, list) and not isinstance(stride, ListConfig):
        # Conv stride only for the final block (with unit stride for all other blocks)
        # to downsample the entire conformer encoder by this factor.
        stride = [1] * (num_layers - 1) + [stride]
    if not isinstance(conv_lpad, list) and not isinstance(conv_lpad, ListConfig):
        # Constant conv lpad for all blocks
        conv_lpad = [conv_lpad] * num_layers
    if not isinstance(attn_lpad, list) and not isinstance(attn_lpad, ListConfig):
        # Constant attn lpad for all blocks
        attn_lpad = [attn_lpad] * num_layers

    assert len(attn_window_size) == num_layers
    assert len(kernel_size) == num_layers
    assert len(stride) == num_layers
    assert len(conv_lpad) == num_layers
    assert len(attn_lpad) == num_layers

    seq: list[nn.Module] = []
    for i in range(num_layers):
        seq.append(
            ConformerEncoderBlock(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                kernel_size=kernel_size[i],
                stride=stride[i],
                num_heads=num_heads,
                attn_window_size=attn_window_size[i],
                attn_lpad=attn_lpad[i],
                dropout=dropout,
            )
        )
    return SlicedSequential(*seq)


def HandwritingConformer(
    in_dim: int,
    out_dim: int,
    input_dim: int,
    ffn_dim: int,
    kernel_size: int | list[int],
    stride: int | list[int],
    num_heads: int,
    attn_window_size: int | list[int],
    log_softmax: bool = True,
    num_layers: int | None = None,
    dropout: float = 0.0,
    time_reduction_stride: int = 1,
):
    """Builder function for a conformer-based handwriting model.

    -> TimeReductionLayer
    -> Linear (If needed)
    -> Conformer block (n layers)
    -> Linear
    -> LogSoftmax
    """

    seq: list[nn.Module] = []
    dim = in_dim
    if time_reduction_stride > 1:
        seq.extend(
            [
                TimeReductionLayer(stride=time_reduction_stride, lpad="none"),
                nn.Linear(dim * time_reduction_stride, input_dim),
            ]
        )
        dim = input_dim

    # Can happen when neither TDS stages nor TimeReductionLayer are added
    if dim != input_dim:
        seq.append(nn.Linear(dim, input_dim))

    seq.extend(
        [
            ConformerEncoder(
                input_dim=input_dim,
                ffn_dim=ffn_dim,
                kernel_size=kernel_size,
                stride=stride,
                num_heads=num_heads,
                attn_window_size=attn_window_size,
                num_layers=num_layers,
                attn_lpad="steady",
                dropout=dropout,
            ),
            nn.Linear(input_dim, out_dim),
            nn.LogSoftmax(dim=-1) if log_softmax else nn.Identity(),
        ]
    )

    return SlicedSequential(*seq)


class HandwritingArchitecture(nn.Module):
    def __init__(
        self,
        num_channels,
        vocab_size,
        featurizer: MultivariatePowerFrequencyFeatures,
        specgram_augment,
        invariance_layer,
        encoder,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.vocab_size = vocab_size

        self.featurizer = featurizer
        self.specaug = specgram_augment
        self.rotation_invariant_mlp = invariance_layer
        self.conformer = encoder

        self.slice = slice(self.conformer.extra_left_context, -1, self.conformer.stride)

    def forward(self, inputs):
                x = self.featurizer(inputs)
        x = self.specaug(x)
        x = self.rotation_invariant_mlp(x)
        x = x.transpose(-1, -2)
        emissions = self.conformer.forward(x)

        return emissions, self.slice

    def compute_time_downsampling(
        self, emg_lengths: torch.Tensor, slc: slice
    ) -> Sequence[int]:

        # Featurization
        emg_lengths = self.featurizer.compute_time_downsampling(emg_lengths)

        emg_lengths = (
            torch.div(emg_lengths - slc.start - 1, slc.step, rounding_mode="trunc") + 1
        )
        return emg_lengths
