"""
Module to load the audio dataset.
"""

import glob
import math
import os
import random

import dac
import torch
import torchaudio
from audiotools import AudioSignal
from audiotools import transforms as tfm
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    """Class to load the audio dataset."""

    def __init__(
        self,
        audio_dir: str,
        max_duration: int = 10,
        mono: bool = True,
        freq: int = 16000,
        transform: list = [
            tfm.CorruptPhase,
            tfm.FrequencyNoise,
            tfm.HighPass,
            tfm.LowPass,
            tfm.MuLawQuantization,
            tfm.NoiseFloor,
            tfm.Quantization,
            tfm.Smoothing,
            tfm.TimeNoise,
        ],
        overall_prob: float = 0.5,
    ):
        """Initializes the dataset.

        Args:
            audio_dir (str): The path to the audio directory.
            max_duration (int): The max duration of the audio in seconds.
            mono (bool): Whether to load the audio as mono.
            input_freq (int): The input frequency of the audio.
            output_freq (int): The output frequency of the audio.
            transform (list): The list of transforms to apply to the audio.
            overall_prob (float): The overall probability of applying the transforms.
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

        self._cut_length = max_duration * freq

        self._mono = mono

        model_path = dac.utils.download(model_type="44khz")
        self.autoencoder = dac.DAC.load(model_path).to("cuda", dtype=torch.float32)
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self._prob = overall_prob / len(transform)
        self._transform = tfm.Compose([trsfm(prob=self._prob) for trsfm in transform])

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

        kwargs = self._transform.instantiate(signal=compressed_waveform.clone())
        compressed_waveform = self._transform(compressed_waveform.clone(), **kwargs)

        min_length = min(base_waveform.shape[-1], compressed_waveform.shape[-1])
        if min_length > self._cut_length:
            min_length = self._cut_length

        compressed_waveform = compressed_waveform[:, :, :min_length].audio_data
        base_waveform = base_waveform[:, :, :min_length].audio_data

        compressed_waveform = compressed_waveform.transpose(0, 1).cuda()
        base_waveform = base_waveform.transpose(0, 1).cuda()

        if self._mono:
            compressed_waveform = compressed_waveform.mean(dim=1)
            base_waveform = base_waveform.mean(dim=1)
        else:
            if compressed_waveform.shape[0] == 1:
                compressed_waveform = compressed_waveform.repeat(2, 1, 1)
            if base_waveform.shape[0] == 1:
                base_waveform = base_waveform.repeat(2, 1, 1)

        compressed_waveform = torch.nn.functional.pad(
            compressed_waveform, (0, self._cut_length - compressed_waveform.shape[-1])
        )
        base_waveform = torch.nn.functional.pad(
            base_waveform, (0, self._cut_length - base_waveform.shape[-1])
        )

        encoded_compressed_waveform, _, _, _, _, _ = self.autoencoder.encode(
            compressed_waveform
        )

        encoded_base_waveform, _, _, _, _, _ = self.autoencoder.encode(base_waveform)
        base_waveform = self.autoencoder.decoder(encoded_base_waveform)

        return (
            encoded_compressed_waveform,
            encoded_base_waveform,
            base_waveform,
        )
