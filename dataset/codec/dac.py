""" 
Encode and decode audio using DAC (descript-audio-codec)
"""

import dac
import torch
from audiotools import AudioSignal

from dataset.codec.codec import Codec


class DAC(Codec):
    """Class that encode the audio"""

    def __init__(self, max_length: int = 180) -> None:
        """Initialize the DAC codec

        Args:
            max_length (int, optional): the max length of an audio. Defaults to 30.
        """

        model_path = dac.utils.download(model_type="16khz")
        self._model = dac.DAC.load(model_path)

        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")

        self._model.to(self._device)
        self._max_length = max_length

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Function to load the audio

        Args:
            audio_path (str): The file path of the audio

        Returns:
            torch.Tensor: The audio tensor
        """

        wav = AudioSignal(audio_path, duration=self._max_length)
        wav.resample(self._model.sample_rate)

        return wav.to(self._device)

    def encoder_decoder(self, audio_path: str, target_path: str) -> None:
        """Encode and decode the audio using DAC

        Args:
            audio_path (str): The file path of the audio
            target_path (str): The file path to save the audio
        """

        signal = self._load_audio(audio_path)
        x = self._model.compress(signal.cpu())
        y = self._model.decompress(x)

        y.write(target_path)

    def __str__(self) -> str:
        return "dac"
