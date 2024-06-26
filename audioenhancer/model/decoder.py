from typing import List

import torch
from torch import nn

from audioenhancer.model.modules import CausalConvTranspose1d, ResidualUnit


class DecoderBlock(nn.Module):
    """Decoder block"""

    def __init__(self, out_channels, stride):
        """
        The Decoder block is composed of a CausalConvTranspose1d layer followed by three ResidualUnit layers.

        Args:
            out_channels (int): The number of output channels
            stride (int): The stride of the convolutional layer
        """
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride,
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=1
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=2
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=3
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """Decoder"""

    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        """
        The Decoder is composed of a series of DecoderBlock layers followed by a convolutional layer.
        It decompresses the audio signal.

        Args:
            C (int): The number of channels
            D (int): The latent space dimension
            strides (tuple, optional): The strides of the convolutional layers. Defaults to (2, 4, 5, 8).
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=D,
                out_channels=16 * C,
                kernel_size=7,
                padding="same",
                padding_mode="reflect",
            ),
            DecoderBlock(out_channels=8 * C, stride=strides[3]),
            DecoderBlock(out_channels=4 * C, stride=strides[2]),
            DecoderBlock(out_channels=2 * C, stride=strides[1]),
            DecoderBlock(out_channels=C, stride=strides[0]),
            nn.Conv1d(
                in_channels=C,
                out_channels=2,
                kernel_size=7,
                padding="same",
                padding_mode="reflect",
            ),
        )

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass"""

        for skip, layer in zip(reversed(skips), self.layers):
            x = layer(x)
            x = x + skip

        return x
