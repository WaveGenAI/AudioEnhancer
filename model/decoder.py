""" 
Decoder block
"""

import torch.nn as nn
from model.units import CausalConv1d, CausalConvTranspose1d, ResidualUnit


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(DecoderBlock, self).__init__()

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
                in_channels=out_channels, out_channels=out_channels, dilation=3
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=9
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(

            CausalConv1d(in_channels=C, out_channels=2, kernel_size=7),
        )

    def forward(self, x):
        return self.layers(x)
