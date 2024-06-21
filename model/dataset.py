import glob
import os

import torch
import torchaudio
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    """Class to load the audio dataset."""

    def __init__(self, audio_dir: str, pad_length: int = 16000 * 30):
        """Initializes the dataset.

        Args:
            audio_dir (str): The path to the audio directory.
            pad_length (int): The length to pad the waveforms to.
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

        _, self.sr = torchaudio.load(self.filenames[0])
        self._pad_length = pad_length

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
        compressed_waveform, _ = torchaudio.load(compressed_file)

        resampler = torchaudio.transforms.Resample(
            sample_rate, 16000, dtype=base_waveform.dtype
        )

        base_waveform = resampler(base_waveform)
        compressed_waveform = resampler(compressed_waveform)

        base_waveform = base_waveform.mean(dim=0, keepdim=True)
        compressed_waveform = compressed_waveform.mean(dim=0, keepdim=True)

        if base_waveform.shape[-1] < self._pad_length:
            base_waveform = torch.nn.functional.pad(
                base_waveform,
                (0, self._pad_length - base_waveform.shape[-1]),
                "constant",
                0,
            )

        if compressed_waveform.shape[-1] < self._pad_length:
            compressed_waveform = torch.nn.functional.pad(
                compressed_waveform,
                (0, self._pad_length - compressed_waveform.shape[-1]),
                "constant",
                0,
            )

        return base_waveform, compressed_waveform
