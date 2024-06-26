from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch import Tensor

"""
Convolutional Modules
"""


def downsample1d(
        in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
    """
    Creates a 1D convolutional layer for downsampling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        factor (int): The downsampling factor.
        kernel_multiplier (int, optional): Multiplier for the kernel size. Defaults to 2.

    Returns:
        nn.Module: The created 1D convolutional layer.
    """
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * kernel_multiplier + 1,
        stride=factor,
        padding=factor * (kernel_multiplier // 2),
    )


def upsample1d(in_channels: int, out_channels: int, factor: int) -> nn.Module:
    """
    Creates a 1D convolutional layer for upsampling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        factor (int): The upsampling factor.

    Returns:
        nn.Module: The created 1D convolutional layer.
    """
    if factor == 1:
        return nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
    return nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * 2,
        stride=factor,
        padding=factor // 2 + factor % 2,
        output_padding=factor % 2,
    )


class ConvBlock1d(nn.Module):
    """
    A convolutional block consisting of a group normalization, an activation function, and a convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        num_groups (int, optional): Number of groups for the group normalization. Defaults to 8.
        use_norm (bool, optional): Whether to use normalization. Defaults to True.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            num_groups: int = 8,
            use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.project = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            use_norm: bool = True,
            num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups,
        )

        self.to_out = (
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)


class Patcher(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super().__init__()
        assert_message = f"out_channels must be divisible by patch_size ({patch_size})"
        assert out_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels // patch_size,
            num_groups=min(patch_size, in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        x = rearrange(x, "b c (l p) -> b (c p) l", p=self.patch_size)
        return x


class Unpatcher(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super().__init__()
        assert_message = f"in_channels must be divisible by patch_size ({patch_size})"
        assert in_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels // patch_size,
            out_channels=out_channels,
            num_groups=min(patch_size, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, " b (c p) l -> b c (l p) ", p=self.patch_size)
        x = self.block(x)
        return x


class DownsampleBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            factor: int,
            num_groups: int,
            num_layers: int,
    ):
        super().__init__()

        self.downsample = downsample1d(
            in_channels=in_channels, out_channels=out_channels, factor=factor
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_groups=num_groups,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class UpsampleBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            factor: int,
            num_layers: int,
            num_groups: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_groups=num_groups,
                )
                for _ in range(num_layers)
            ]
        )

        self.upsample = upsample1d(
            in_channels=in_channels, out_channels=out_channels, factor=factor
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.upsample(x)
        return x


"""
Encoders / Decoders
"""


class Bottleneck(nn.Module):
    def forward(
            self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        raise NotImplementedError()


"""
Bottlenecks
"""


def gaussian_sample(mean: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    sample = mean + std * eps
    return sample


def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
    losses = mean ** 2 + logvar.exp() - logvar - 1
    loss = reduce(losses, "b ... -> 1", "mean").item()
    return loss


class VariationalBottleneck(Bottleneck):
    def __init__(self, channels: int, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.to_mean_and_std = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
        )

    def forward(
            self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        mean_and_std = self.to_mean_and_std(x)
        mean, std = mean_and_std.chunk(chunks=2, dim=1)
        mean = torch.tanh(mean)  # mean in range [-1, 1]
        std = torch.tanh(std) + 1.0  # std in range [0, 2]
        out = gaussian_sample(mean, std)
        info = dict(
            variational_kl_loss=kl_loss(mean, std) * self.loss_weight,
            variational_mean=mean,
            variational_std=std,
        )
        return (out, info) if with_info else out


class TanhBottleneck(Bottleneck):
    def forward(
            self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        x = torch.tanh(x)
        info: Dict = dict()
        return (x, info) if with_info else x


class NoiserBottleneck(Bottleneck):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
            self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        if self.training:
            x = torch.randn_like(x) * self.sigma + x
        info: Dict = dict()
        return (x, info) if with_info else x
