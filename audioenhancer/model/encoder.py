"""
Encoder block
"""

import torch.nn as nn

from audioenhancer.model.modules import CausalConv1d, ResidualUnit


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(EncoderBlock, self).__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=1,
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=3,
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=9,
            ),
            CausalConv1d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(Encoder, self).__init__()

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

    def forward(self, x):
        return self.layers(x)
