# Created on Thu May 23 2024
#
# The MIT License (MIT)
# Copyright (c) 2024 WaveAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

""" 
Generate the encoded audio
"""

import io

import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

from data.codec import codec


class Encodec(codec.Codec):
    """Class that encode the audio"""

    def __init__(self, bandewidth: int = 6) -> None:
        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(bandewidth)

    def _load_audio(self, audio_path: str):
        wav, sr = torchaudio.load(audio_path)
        wav = wav.unsqueeze(0)
        wav = convert_audio(wav, sr, self._model.sample_rate, self._model.channels)

        return wav

    def encode(self, audio_path: str) -> io.BytesIO:
        """Encode the audio and return a BytesIO object

        Args:
            audio_path (str): The file path of the audio

        Returns:
            io.BytesIO: The BytesIO object of the encoded audio
        """

        audio = self._load_audio(audio_path)
        encoded_audio = self._model.encode(audio)
        decodec_audio = self._model.decode(encoded_audio)

        return decodec_audio
