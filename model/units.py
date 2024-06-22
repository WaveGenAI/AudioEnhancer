"""
Specific units used in the model.
"""

import torch.nn as nn
import torch.nn.functional as F


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


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad_mode='reflect', **kwargs):
        super(CausalConv1d, self).__init__()

        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)

        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, [self.causal_padding, 0], mode=self.pad_mode)
        return self.conv(x)
    
    

class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(CausalConvTranspose1d, self).__init__()

        self.upsample_factor = stride
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            **kwargs
        )

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        return out[..., :(n * self.upsample_factor)]
