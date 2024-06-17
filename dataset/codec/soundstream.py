""" 
Encode and decode audio using SoundStream
"""

import torch
import torchaudio
from soundstream import from_pretrained, load

from dataset.codec.codec import Codec


class Soundstream(Codec):
    """Class that encode the audio"""

    def __init__(self) -> None:
        self._model = from_pretrained()

        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")

        self._model.to(self._device)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Function to load the audio

        Args:
            audio_path (str): The file path of the audio

        Returns:
            torch.Tensor: The audio tensor
        """

        wav = load(audio_path)
        return wav.to(self._device)

    def encoder_decoder(self, audio_path: str, target_path: str) -> None:
        """Encode and decode the audio using Soundstream

        Args:
            audio_path (str): The file path of the audio
            target_path (str): The file path to save the audio
        """

        print(f"Encoding {audio_path} to {target_path} with Soundstream")

        audio = self._load_audio(audio_path)

        with torch.no_grad():
            quantized = self._model(audio, mode="encode")
            recovered = self._model(quantized, mode="decode")

        torchaudio.save(target_path, recovered[0].cpu(), 16000)

    def __str__(self) -> str:
        return "soundstream"
