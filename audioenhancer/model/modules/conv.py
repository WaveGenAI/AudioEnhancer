""" 
Convolutional neural network module.
"""

from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal convolutional layer."""
    def __init__(
        self, in_channels, out_channels, kernel_size, pad_mode="reflect", **kwargs
    ):
        """
        Causal convolutional layer.

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            kernel_size (int): The kernel size
            pad_mode (str, optional): The padding mode. Defaults to "reflect".
        """
        super(CausalConv1d, self).__init__()

        dilation = kwargs.get("dilation", 1)
        stride = kwargs.get("stride", 1)

        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, [self.causal_padding, 0], mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """Causal convolutional transpose layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        """
        Causal convolutional transpose layer.
        """
        super(CausalConvTranspose1d, self).__init__()

        self.upsample_factor = stride
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, **kwargs
        )

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        return out[..., : (n * self.upsample_factor)]
