"""
Residual unit module.
"""

import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, pad_mode="reflect"):
        super(ResidualUnit, self).__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=dilation * 3,
                padding_mode=pad_mode,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding_mode=pad_mode,
            ),
            nn.ELU(),
        )

    def forward(self, x):
        return x + self.layers(x)
