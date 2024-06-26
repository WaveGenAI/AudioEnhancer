"""Decoder for 1D audio autoencoder."""

from typing import Any, Optional, Sequence, Tuple, Union, List

from torch import nn, Tensor

from audioenhancer.model.audio_ae.modules import UpsampleBlock1d, Unpatcher
from audioenhancer.model.audio_ae.utils import exists

class Decoder1d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        patch_size: int = 1,
        resnet_groups: int = 8,
        in_channels: Optional[int] = None,
    ):
        super().__init__()
        num_layers = len(multipliers) - 1

        assert len(factors) == num_layers and len(num_blocks) == num_layers

        self.to_in = (
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=channels * multipliers[0],
                kernel_size=1,
            )
            if exists(in_channels)
            else nn.Identity()
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    factor=factors[i],
                    num_groups=resnet_groups,
                    num_layers=num_blocks[i],
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = Unpatcher(
            in_channels=channels * multipliers[-1],
            out_channels=out_channels,
            patch_size=patch_size,
        )

    def forward(
        self, x: Tensor, encoder_info: List[Tensor], with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        encoder_info.reverse()
        x = x + encoder_info[0]
        x = self.to_in(x)
        xs = [x]

        for i, upsample in enumerate(self.upsamples):
            x = x + encoder_info[i + 1]
            x = upsample(x)
            xs += [x]

        x = x + encoder_info[-1]
        x = self.to_out(x)
        xs += [x]

        info = dict(xs=xs)
        return (x, info) if with_info else x