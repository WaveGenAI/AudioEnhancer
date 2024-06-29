"""
Module to load the audio dataset.
"""

import glob
import math
import os

import torch
import torchaudio
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    """Class to load the audio dataset."""

    def __init__(
        self,
        audio_dir: str,
        max_duration: int = 10,
        mono: bool = True,
        input_freq: int = 16000,
        output_freq: int = 16000,
    ):
        """Initializes the dataset.

        Args:
            audio_dir (str): The path to the audio directory.
            max_duration (int): The max duration of the audio in seconds.
            mono (bool): Whether to load the audio as mono.
            input_freq (int): The input frequency of the audio.
            output_freq (int): The output frequency of the audio.
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

        self._pad_length_input = math.ceil(math.log2(max_duration * input_freq))
        self._pad_length_output = math.ceil(math.log2(max_duration * output_freq))

        self._mono = mono
        self._input_freq = input_freq
        self._output_freq = output_freq

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

        base_waveform, sample_rate = torchaudio.load(base_file)
        compressed_waveform, compress_sr = torchaudio.load(compressed_file)

        base_waveform = torchaudio.transforms.Resample(
            sample_rate, self._output_freq, dtype=base_waveform.dtype
        )(base_waveform)

        compressed_waveform = torchaudio.transforms.Resample(
            compress_sr, self._input_freq, dtype=compressed_waveform.dtype
        )(compressed_waveform)

        if self._mono:
            base_waveform = base_waveform.mean(dim=0, keepdim=True)
            compressed_waveform = compressed_waveform.mean(dim=0, keepdim=True)
        else:
            if base_waveform.shape[0] == 1:
                base_waveform = base_waveform.repeat(2, 1)
            if compressed_waveform.shape[0] == 1:
                compressed_waveform = compressed_waveform.repeat(2, 1)

        if base_waveform.shape[-1] < self._pad_length_output:
            base_waveform = torch.nn.functional.pad(
                base_waveform,
                (0, self._pad_length_output - base_waveform.shape[-1]),
                "constant",
                0,
            )

        if compressed_waveform.shape[-1] < self._pad_length_input:
            compressed_waveform = torch.nn.functional.pad(
                compressed_waveform,
                (0, self._pad_length_input - compressed_waveform.shape[-1]),
                "constant",
                0,
            )

        return (
            compressed_waveform[:, : 2**self._pad_length_input],
            base_waveform[:, : 2**self._pad_length_output],
        )
