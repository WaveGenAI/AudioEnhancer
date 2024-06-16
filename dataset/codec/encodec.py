""" 
Encode and decode audio using Encodec
"""

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio, save_audio

from dataset.codec.codec import Codec


class Encodec(Codec):
    """Class that encode the audio"""

    def __init__(self, bandewidth: int = 6) -> None:
        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(bandewidth)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Function to load the audio

        Args:
            audio_path (str): The file path of the audio

        Returns:
            torch.Tensor: The audio tensor
        """

        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self._model.sample_rate, self._model.channels)

        return wav

    def encoder_decoder(self, audio_path: str, target_path: str) -> None:
        """Encode and decode the audio using Encodec

        Args:
            audio_path (str): The file path of the audio
            target_path (str): The file path to save the audio
        """

        print(f"Encoding {audio_path} to {target_path} with Encodec")

        audio = self._load_audio(audio_path)

        with torch.no_grad():
            compressed = self._model.encode(audio[None])
            decompressed = self._model.decode(compressed)

            decompressed = decompressed[0, :, : audio.shape[-1]]

        save_audio(decompressed, target_path, self._model.sample_rate)

    def __str__(self) -> str:
        return "encodec"
