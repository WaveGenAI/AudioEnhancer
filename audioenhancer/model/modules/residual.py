"""
Residual unit module.
"""

from torch import nn


class ResidualUnit(nn.Module):
    """
    Residual unit module.
    """
    def __init__(self, in_channels, out_channels, dilation, pad_mode="reflect"):
        """
        Residual unit module.

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            dilation (int): The dilation factor
            pad_mode (str, optional): The padding mode. Defaults to "reflect".
        """
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
            nn.ReLU(),
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
