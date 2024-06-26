"""Encoder module for 1D audio data."""

from torch import nn, Tensor
from typing import Any, Optional, Sequence, Tuple, Union, List

from audioenhancer.model.audio_ae.modules import Bottleneck, DownsampleBlock1d, Patcher
from audioenhancer.model.audio_ae.utils import to_list, exists, prefix_dict, prod

class Encoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        patch_size: int = 1,
        resnet_groups: int = 8,
        out_channels: Optional[int] = None,
        bottleneck: Union[Bottleneck, List[Bottleneck]] = [],
    ):
        super().__init__()
        self.bottlenecks = nn.ModuleList(to_list(bottleneck))
        self.num_layers = len(multipliers) - 1
        self.downsample_factor = patch_size * prod(factors)
        self.out_channels = (
            out_channels if exists(out_channels) else channels * multipliers[-1]
        )
        assert len(factors) == self.num_layers and len(num_blocks) == self.num_layers

        self.to_in = Patcher(
            in_channels=in_channels,
            out_channels=channels * multipliers[0],
            patch_size=patch_size,
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    factor=factors[i],
                    num_groups=resnet_groups,
                    num_layers=num_blocks[i],
                )
                for i in range(self.num_layers)
            ]
        )

        self.to_out = (
            nn.Conv1d(
                in_channels=channels * multipliers[-1],
                out_channels=out_channels,
                kernel_size=1,
            )
            if exists(out_channels)
            else nn.Identity()
        )

    def forward(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        x = self.to_in(x)
        xs = [x]

        for downsample in self.downsamples:
            x = downsample(x)
            xs += [x]

        x = self.to_out(x)
        xs += [x]
        info = dict(xs=xs)

        for bottleneck in self.bottlenecks:
            x, info_bottleneck = bottleneck(x, with_info=True)
            info = {**info, **prefix_dict("bottleneck_", info_bottleneck)}

        return (x, info) if with_info else x
