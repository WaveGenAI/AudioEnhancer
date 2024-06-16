""" 
Module that contain an abstract class for codec
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
    def decode(self, audio: io.BytesIO) -> io.BytesIO:
        """Decode the audio and return a BytesIO object

        Args:
            audio (io.BytesIO): The BytesIO object of the audio

        Returns:
            io.BytesIO: The BytesIO object of the decoded audio
        """
