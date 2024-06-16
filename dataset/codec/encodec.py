""" 
Encode and decode audio using Encodec
"""

from io import BytesIO
from dataset.codec.codec import Codec


class Encodec(Codec):
    """Module to encode audio files with encodec."""

    def encode(self, audio_path: str) -> BytesIO:
        raise NotImplementedError

    def decode(self, audio: BytesIO) -> BytesIO:
        raise NotImplementedError

    def __str__(self):
        return "encodec"
