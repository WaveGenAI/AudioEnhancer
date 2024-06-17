""" 
Encode and decode audio using Opus
"""

import tempfile

import ffmpeg

from dataset.codec.codec import Codec


class Opus(Codec):
    """Class that encode the audio"""

    def __init__(self, bandewidth: int = 16) -> None:
        """Initialize the Opus codec

        Args:
            bandewidth (int, optional): the bitrate of the audio. Defaults to 16.
        """

        self._bandwidth = f"{bandewidth}k"

    def encoder_decoder(self, audio_path: str, target_path: str) -> None:
        """Encode and decode the audio using Opus

        Args:
            audio_path (str): The file path of the audio
            target_path (str): The file path to save the audio
        """

        with tempfile.NamedTemporaryFile(suffix=".opus") as temp_audio:
            ffmpeg.input(audio_path).output(
                temp_audio.name, **{"c:a": "libopus", "b:a": self._bandwidth}
            ).run(overwrite_output=True, quiet=True)

            ffmpeg.input(temp_audio.name).output(
                target_path, acodec="libmp3lame", audio_bitrate=self._bandwidth
            ).run(overwrite_output=True, quiet=True)

    def __str__(self) -> str:
        return "opus"
