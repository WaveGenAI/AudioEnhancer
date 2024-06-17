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

    def __init__(self, bandewidth: int = 6, max_length: int = 180) -> None:
        """Initialize the Encodec model

        Args:
            bandewidth (int, optional): the target bandewidth. Defaults to 6.
            max_length (int, optional): the max duration of the audio. Defaults to 180.
        """

        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(bandewidth)

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

        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self._model.sample_rate, self._model.channels)

        # fix max audio length to self._max_length seconds (avoid memory issues)
        if wav.shape[-1] > self._max_length * self._model.sample_rate:
            wav = wav[:, : self._max_length * self._model.sample_rate]

        return wav[None].to(self._device)

    def encoder_decoder(self, audio_path: str, target_path: str) -> None:
        """Encode and decode the audio using Encodec

        Args:
            audio_path (str): The file path of the audio
            target_path (str): The file path to save the audio
        """

        audio = self._load_audio(audio_path)

        with torch.no_grad():
            compressed = self._model.encode(audio)
            decompressed = self._model.decode(compressed)

            decompressed = decompressed[0, :, : audio.shape[-1]]

        save_audio(decompressed.cpu(), target_path, self._model.sample_rate)

    def __str__(self) -> str:
        return "encodec"
