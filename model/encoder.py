"""
Encoder block
"""

import torch.nn as nn
from model.units import CausalConv1d, ResidualUnit


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
            CausalConv1d(in_channels=2, out_channels=C, kernel_size=7),
        )

    def forward(self, x):
        return self.layers(x)
