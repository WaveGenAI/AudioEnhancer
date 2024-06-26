"""
Encoder block
"""
import torch
import torch.nn as nn

from audioenhancer.model.modules import CausalConv1d, ResidualUnit


class EncoderBlock(nn.Module):
    """
    Encoder block
    The Encoder block is composed of a series of ResidualUnit layers followed by a CausalConv1d layer.

    Args:
        out_channels (int): The number of output channels
        stride (int): The stride for the convolutional layer
    """
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=1,
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=2,
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=3,
            ),
            CausalConv1d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.layers(x)


class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        """
        The Encoder is composed of a series of EncoderBlock layers followed by a convolutional layer.
        It compresses the audio signal.

        Args:
            C (int): The number of channels
            D (int): The latent space dimension
            strides (tuple, optional): The strides of the convolutional layers. Defaults to (2, 4, 5, 8).
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=C,
                kernel_size=7,
                padding="same",
                padding_mode="reflect",
            ),
            EncoderBlock(out_channels=2 * C, stride=strides[0]),
            EncoderBlock(out_channels=4 * C, stride=strides[1]),
            EncoderBlock(out_channels=8 * C, stride=strides[2]),
            EncoderBlock(out_channels=16 * C, stride=strides[3]),
            nn.Conv1d(
                in_channels=16 * C,
                out_channels=D,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.layers(x)
