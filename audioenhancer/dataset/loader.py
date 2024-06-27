"""
Module to load the audio dataset.
"""

import glob
import os

import torch
import torchaudio
from torch.utils.data import Dataset

from audioenhancer.constants import SAMPLING_RATE, UPSAMPLE_RATE


class SynthDataset(Dataset):
    """Class to load the audio dataset."""

    def __init__(
        self, audio_dir: str, max_duration: int = 10, mono: bool = True
    ):
        """Initializes the dataset.

        Args:
            audio_dir (str): The path to the audio directory.
            max_duration (int): The max duration of the audio in seconds.
            mono (bool): Whether to load the audio as mono.
        """

        super().__init__()

        self.filenames = glob.glob(audio_dir + "/*.mp3")

        self.codecs = [
            f
            for f in glob.glob(
                audio_dir + "/*",
            )
            if os.path.isdir(f)
        ]

        self._pad_length = int(max_duration * SAMPLING_RATE)
        self._mono = mono

    def __len__(self) -> int:
        """Returns the number of waveforms in the dataset.

        Returns:
            int: Number of waveforms in the dataset.
        """

        return len(self.filenames) * len(self.codecs)

    def __getitem__(self, index: int) -> tuple:
        """Fetches the waveform for the given index.

        Args:
            index (int): Index of the waveform to fetch.

        Returns:
            tuple: A tuple containing the base waveform and the compressed waveform.
        """

        base_idx = index // len(self.codecs)
        codec_idx = index % len(self.codecs)

        codec = self.codecs[codec_idx]

        base_file = self.filenames[base_idx]
        compressed_file = os.path.join(codec, os.path.basename(base_file))

        original_waveform, sample_rate = torchaudio.load(base_file)
        compressed_waveform, compress_sr = torchaudio.load(compressed_file)

        base_waveform = torchaudio.transforms.Resample(
            sample_rate, SAMPLING_RATE, dtype=original_waveform.dtype
        )(original_waveform)

        upsample_base = torchaudio.transforms.Resample(
            sample_rate, UPSAMPLE_RATE, dtype=original_waveform.dtype
        )(original_waveform)

        compressed_waveform = torchaudio.transforms.Resample(
            compress_sr, SAMPLING_RATE, dtype=compressed_waveform.dtype
        )(compressed_waveform)


        if self._mono:
            base_waveform = base_waveform.mean(dim=0, keepdim=True)
            compressed_waveform = compressed_waveform.mean(dim=0, keepdim=True)
        else:
            if base_waveform.shape[0] == 1:
                base_waveform = base_waveform.repeat(2, 1)
            if compressed_waveform.shape[0] == 1:
                compressed_waveform = compressed_waveform.repeat(2, 1)
            if upsample_base.shape[0] == 1:
                upsample_base = upsample_base.repeat(2, 1)

        if base_waveform.shape[-1] < self._pad_length:
            base_waveform = torch.nn.functional.pad(
                base_waveform,
                (0, self._pad_length - base_waveform.shape[-1]),
                "constant",
                0,
            )
        else:
            base_waveform = base_waveform[:, : self._pad_length]

        if compressed_waveform.shape[-1] < self._pad_length:
            compressed_waveform = torch.nn.functional.pad(
                compressed_waveform,
                (0, self._pad_length - compressed_waveform.shape[-1]),
                "constant",
                0,
            )
        else:
            compressed_waveform = compressed_waveform[:, : self._pad_length]
        up_pad_length = self._pad_length * UPSAMPLE_RATE // SAMPLING_RATE
        if upsample_base.shape[-1] < up_pad_length:
            upsample_base = torch.nn.functional.pad(
                upsample_base,
                (0, up_pad_length - upsample_base.shape[-1]),
                "constant",
                0,
            )
        else:
            upsample_base = upsample_base[:, : up_pad_length]

        return compressed_waveform, base_waveform, upsample_base
