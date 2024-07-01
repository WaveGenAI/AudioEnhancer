"""
Module to load the audio dataset.
"""

import glob
import math
import os

import dac
import torch
import torchaudio
from audiotools import AudioSignal
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

        model_path = dac.utils.download(model_type="44khz")
        self.autoencoder = dac.DAC.load(model_path).to("cuda")
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)

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

        base_waveform = AudioSignal(base_file)
        compressed_waveform = AudioSignal(compressed_file)

        base_waveform = base_waveform.resample(self.autoencoder.sample_rate)
        compressed_waveform = compressed_waveform.resample(self.autoencoder.sample_rate)

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

        compressed_waveform = compressed_waveform[:, : 2**self._pad_length_input]
        base_waveform = base_waveform[:, : 2**self._pad_length_output]

        encoded_compressed_waveform = self.autoencoder.compress(
            compressed_waveform
        ).codes
        encoded_base_waveform = self.autoencoder.compress(base_waveform).codes

        # convert to mono or stereo
        if self._mono:
            encoded_compressed_waveform = encoded_compressed_waveform.mean(dim=1)
            encoded_base_waveform = encoded_base_waveform.mean(dim=1)

        if not self._mono:
            if encoded_compressed_waveform.shape[0] == 1:
                encoded_compressed_waveform = encoded_compressed_waveform.repeat(
                    2, 1, 1
                )
            if encoded_base_waveform.shape[0] == 1:
                encoded_base_waveform = encoded_base_waveform.repeat(2, 1, 1)

        return (
            encoded_compressed_waveform,
            encoded_base_waveform,
        )
