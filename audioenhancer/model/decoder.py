import torch.nn as nn

from audioenhancer.model.modules import CausalConvTranspose1d, ResidualUnit


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
                in_channels=out_channels, out_channels=out_channels, dilation=2
            ),
            ResidualUnit(
                in_channels=out_channels, out_channels=out_channels, dilation=3
            ),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(Decoder, self).__init__()

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

    def forward(self, x):
        return self.layers(x)
