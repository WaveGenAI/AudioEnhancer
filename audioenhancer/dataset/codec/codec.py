""" 
Module that contain an abstract class for codec
"""

from abc import abstractmethod


class Codec:
    """Abstract class for codec"""

    @abstractmethod
    def encoder_decoder(self, audio_path: str, target_path: str) -> None:
        """Encode and decode the audio

        Args:
            audio_path (str): The file path of the audio
            target_path (str): The file path to save the audio
        """

    @abstractmethod
    def __str__(self) -> str:
        """Return the name of the codec"""
