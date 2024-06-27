"""This module contains all the auto-encoder models for audio data."""

from math import floor
from typing import Any, Optional, Sequence, Tuple, Union, List, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchaudio import transforms

from einops_exts import rearrange_many
from einops import pack, rearrange, unpack

from audioenhancer.model.audio_ae.latent import LatentProcessor
from audioenhancer.model.audio_ae.modules import Bottleneck
from audioenhancer.model.audio_ae.utils import (
    prefix_dict,
    default,
    groupby,
    closest_power_2,
)
from audioenhancer.model.audio_ae.encoder import Encoder1d
from audioenhancer.model.audio_ae.decoder import Decoder1d


class AutoEncoder1d(nn.Module):
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
        bottleneck_channels: Optional[int] = None,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)

        self.encoder = Encoder1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            channels=channels,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            patch_size=patch_size,
            resnet_groups=resnet_groups,
            bottleneck=bottleneck,
        )

        self.latent = LatentProcessor(512, 8)

        self.decoder = Decoder1d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            channels=channels,
            multipliers=multipliers[::-1],
            factors=factors[::-1],
            num_blocks=num_blocks[::-1],
            patch_size=patch_size,
            resnet_groups=resnet_groups,
        )

    def forward(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        z, info_encoder = self.encode(x, with_info=True)

        y, info_decoder = self.decode(z, info_encoder["xs"], with_info=True)
        info = {
            **dict(latent=z),
            **prefix_dict("encoder_", info_encoder),
            **prefix_dict("decoder_", info_decoder),
        }
        return (y, info) if with_info else y

    def encode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        return self.encoder(x, with_info=with_info)

    def decode(
        self, x: Tensor, encoder_info: List[Tensor], with_info: bool = False
    ) -> Tensor:
        return self.decoder(x, encoder_info, with_info=with_info)


class STFT(nn.Module):
    """Helper for torch stft and istft"""

    def __init__(
        self,
        num_fft: int = 1023,
        hop_length: int = 256,
        window_length: Optional[int] = None,
        length: Optional[int] = None,
        use_complex: bool = False,
    ):
        super().__init__()
        self.num_fft = num_fft
        self.hop_length = default(hop_length, floor(num_fft // 4))
        self.window_length = default(window_length, num_fft)
        self.length = length
        self.register_buffer("window", torch.hann_window(self.window_length))
        self.use_complex = use_complex

    def encode(self, wave: Tensor) -> Tuple[Tensor, Tensor]:
        b = wave.shape[0]
        wave = rearrange(wave, "b c t -> (b c) t")

        stft = torch.stft(
            wave,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            return_complex=True,
            normalized=True,
        )

        if self.use_complex:
            # Returns real and imaginary
            stft_a, stft_b = stft.real, stft.imag
        else:
            # Returns magnitude and phase matrices
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            stft_a, stft_b = magnitude, phase

        return rearrange_many((stft_a, stft_b), "(b c) f l -> b c f l", b=b)

    def decode(self, stft_a: Tensor, stft_b: Tensor) -> Tensor:
        b, l = stft_a.shape[0], stft_a.shape[-1]  # noqa
        length = closest_power_2(l * self.hop_length)

        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> (b c) f l")

        if self.use_complex:
            real, imag = stft_a, stft_b
        else:
            magnitude, phase = stft_a, stft_b
            real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)

        stft = torch.stack([real, imag], dim=-1)

        wave = torch.istft(
            stft,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            length=default(self.length, length),
            normalized=True,
        )

        return rearrange(wave, "(b c) t -> b c t", b=b)

    def encode1d(
        self, wave: Tensor, stacked: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        stft_a, stft_b = self.encode(wave)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b c f l -> b (c f) l")
        return torch.cat((stft_a, stft_b), dim=1) if stacked else (stft_a, stft_b)

    def decode1d(self, stft_pair: Tensor) -> Tensor:
        f = self.num_fft // 2 + 1
        stft_a, stft_b = stft_pair.chunk(chunks=2, dim=1)
        stft_a, stft_b = rearrange_many((stft_a, stft_b), "b (c f) l -> b c f l", f=f)
        return self.decode(stft_a, stft_b)


class ME1d(Encoder1d):
    """Magnitude Encoder"""

    def __init__(
        self, in_channels: int, stft_num_fft: int, use_log: bool = False, **kwargs
    ):
        self.use_log = use_log
        self.frequency_channels = stft_num_fft // 2 + 1
        stft_kwargs, kwargs = groupby("stft_", kwargs)
        super().__init__(in_channels=in_channels * self.frequency_channels, **kwargs)
        self.stft = STFT(num_fft=stft_num_fft, **stft_kwargs)
        self.downsample_factor *= self.stft.hop_length

    def forward(self, x: Tensor, **kwargs) -> Union[Tensor, Tuple[Tensor, Any]]:  # type: ignore # noqa
        magnitude, _ = self.stft.encode(x)
        magnitude = rearrange(magnitude, "b c f l -> b (c f) l")
        magnitude = torch.log(magnitude) if self.use_log else magnitude
        return super().forward(magnitude, **kwargs)


class MAE1d(AutoEncoder1d):
    """Magnitude Auto Encoder"""

    def __init__(self, in_channels: int, stft_num_fft: int = 1023, **kwargs):
        self.frequency_channels = stft_num_fft // 2 + 1
        stft_kwargs, kwargs = groupby("stft_", kwargs)
        super().__init__(in_channels=in_channels * self.frequency_channels, **kwargs)
        self.stft = STFT(num_fft=stft_num_fft, **stft_kwargs)

    def encode(self, magnitude: Tensor, **kwargs):  # type: ignore
        log_magnitude = torch.log(magnitude)
        log_magnitude_flat = rearrange(log_magnitude, "b c f l -> b (c f) l")
        return super().encode(log_magnitude_flat, **kwargs)

    def decode(  # type: ignore
        self, latent: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        f = self.frequency_channels
        log_magnitude_flat, info = super().decode(latent, with_info=True)
        log_magnitude = rearrange(log_magnitude_flat, "b (c f) l -> b c f l", f=f)
        log_magnitude = torch.clamp(log_magnitude, -30.0, 20.0)
        magnitude = torch.exp(log_magnitude)
        info = dict(log_magnitude=log_magnitude, **info)
        return (magnitude, info) if with_info else magnitude

    def loss(
        self, wave: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        magnitude, _ = self.stft.encode(wave)
        magnitude_pred, info = self(magnitude, with_info=True)
        loss = F.l1_loss(torch.log(magnitude), torch.log(magnitude_pred))
        return (loss, info) if with_info else loss


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sample_rate: int = 48000,
        n_mel_channels: int = 80,
        center: bool = False,
        normalize: bool = False,
        normalize_log: bool = False,
    ):
        super().__init__()
        self.padding = (n_fft - hop_length) // 2
        self.normalize = normalize
        self.normalize_log = normalize_log
        self.hop_length = hop_length

        self.to_spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            power=None,
        )

        self.to_mel_scale = transforms.MelScale(
            n_mels=n_mel_channels, n_stft=n_fft // 2 + 1, sample_rate=sample_rate
        )

    def forward(self, waveform: Tensor) -> Tensor:
        # Pack non-time dimension
        waveform, ps = pack([waveform], "* t")
        # Pad waveform
        waveform = F.pad(waveform, [self.padding] * 2, mode="reflect")
        # Compute STFT
        spectrogram = self.to_spectrogram(waveform)
        # Compute magnitude
        spectrogram = torch.abs(spectrogram)
        # Convert to mel scale
        mel_spectrogram = self.to_mel_scale(spectrogram)
        # Normalize
        if self.normalize:
            mel_spectrogram = mel_spectrogram / torch.max(mel_spectrogram)
            mel_spectrogram = 2 * torch.pow(mel_spectrogram, 0.25) - 1
        if self.normalize_log:
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        # Unpack non-spectrogram dimension
        return unpack(mel_spectrogram, ps, "* f l")[0]


class MelE1d(Encoder1d):
    """Magnitude Encoder"""

    def __init__(self, in_channels: int, mel_channels: int, **kwargs):
        mel_kwargs, kwargs = groupby("mel_", kwargs)
        super().__init__(in_channels=in_channels * mel_channels, **kwargs)
        self.mel = MelSpectrogram(n_mel_channels=mel_channels, **mel_kwargs)
        self.downsample_factor *= self.mel.hop_length

    def forward(self, x: Tensor, **kwargs) -> Union[Tensor, Tuple[Tensor, Any]]:  # type: ignore # noqa
        mel = rearrange(self.mel(x), "b c f l -> b (c f) l")
        return super().forward(mel, **kwargs)
