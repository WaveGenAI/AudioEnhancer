""" 
Encode and decode audio using SoundStream
"""

import torch
import torchaudio
from soundstream import from_pretrained, load

from dataset.codec.codec import Codec


class Soundstream(Codec):
    """Class that encode the audio"""

    def __init__(self, max_length: int = 180) -> None:
        """Initialize the Soundstream model

        Args:
            max_length (int, optional): the max duration of an audio. Defaults to 180.
        """

        self._model = from_pretrained()

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

        wav = load(audio_path)

        # fix max audio length to self._max_length minutes (avoid memory issues)
        if wav.shape[-1] > self._max_length * 16000:
            wav = wav[:, : self._max_length * 16000]

        return wav.to(self._device)

    def encoder_decoder(self, audio_path: str, target_path: str) -> None:
        """Encode and decode the audio using Soundstream

        Args:
            audio_path (str): The file path of the audio
            target_path (str): The file path to save the audio
        """

        audio = self._load_audio(audio_path)

        with torch.no_grad():
            quantized = self._model(audio, mode="encode")
            recovered = self._model(quantized, mode="decode")

        torchaudio.save(target_path, recovered[0].cpu(), 16000)

    def __str__(self) -> str:
        return "soundstream"
