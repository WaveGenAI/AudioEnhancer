#
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
Module that contain a abstract class for codec
"""

import io
from abc import abstractmethod


class Codec:
    """Abstract class for codec"""

    @abstractmethod
    def encode(self, audio_path: str) -> io.BytesIO:
        """Encode the audio and return a BytesIO object

        Args:
            audio_path (str): The file name of the audio

        Returns:
            io.BytesIO: The BytesIO object of the encoded audio
        """

    @abstractmethod
    def _load_audio(self, audio_path: str):
        """Load the audio for the audio codec.

        Args:
            audio_path (str): The path of the audio file
        """

    def save(self, audio_path: str, audio: io.BytesIO) -> None:
        """Save the audio

        Args:
            audio_path (str): The file name of the audio
            audio (io.BytesIO): The BytesIO object of the audio
        """

        with open(audio_path, "wb") as f:
            f.write(audio.getvalue())
