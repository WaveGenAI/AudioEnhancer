"""
Code for inference.
"""

import os

import dac
import torch
import torchaudio
from einops import rearrange

from audioenhancer.model.audio_ae.model import model_xtransformer_small as model


class Inference:
    def __init__(self, model_path: str, sampling_rate: int):
        self.model = model
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self._sampling_rate = sampling_rate

        autoencoder_path = dac.utils.download(model_type="44khz")
        self._autoencoder = dac.DAC.load(autoencoder_path).to(self.device)

    def load(self, waveform_path):
        """
        Load the waveform from the given path and resample it to the desired sampling rate.

        Args:
            waveform_path (str): The path to the waveform file.
        """

        waveform, sample_rate = torchaudio.load(waveform_path)
        resampler = torchaudio.transforms.Resample(
            sample_rate, self._sampling_rate, dtype=waveform.dtype
        )
        waveform = resampler(waveform)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        waveform = waveform.unsqueeze(0)

        return waveform.to(self.device)

    def inference(self, audio_path: str, chunk_duration: int = 10):
        """Run inference on the given audio file.

        Args:
            audio_path (str): The path to the audio file to enhance.
            chunk_duration (int): The duration of each chunk in seconds.
        """
        audio = self.load(audio_path)
        output = torch.Tensor().to(self.device)
        ae_input = torch.Tensor().to(self.device)

        chunck_size = self._sampling_rate * chunk_duration
        for i in range(0, audio.size(2), int(chunck_size)):
            chunk = audio[:, :, i : i + int(chunck_size)]

            if chunk.size(2) < int(chunck_size):
                chunk = torch.nn.functional.pad(
                    chunk,
                    (0, int(chunck_size) - chunk.size(2)),
                    "constant",
                    0,
                )

            with torch.no_grad():
                encoded, encoded_q, _, _, _, _ = self._autoencoder.encode(
                    chunk.transpose(0, 1)
                )

                # create input for the model
                decoded = self._autoencoder.decode(encoded_q)

                decoded = decoded.transpose(0, 1)

                ae_input = torch.cat([ae_input, decoded], dim=2)

                encoded = encoded.unsqueeze(0)
                c, d = encoded.shape[1], encoded.shape[2]
                encoded = rearrange(encoded, "b c d t -> b (t c) d")

                pred = self.model(encoded)

                pred = rearrange(pred, "b (t c) d -> b c d t", c=c, d=d)
                pred = pred.squeeze(0)

                # quantize
                z_q, _, _, _, _ = self._autoencoder.quantizer(pred, None)

                decoded = self._autoencoder.decode(z_q)

                decoded = decoded.transpose(0, 1)

                output = torch.cat([output, decoded], dim=2)

        # fix runtime error: numpy
        output = output.squeeze(0).detach().cpu()
        ae_input = ae_input.squeeze(0).detach().cpu()

        torchaudio.save(
            "./data/input.mp3",
            ae_input.T,
            self._sampling_rate,
            channels_first=False,
        )
        torchaudio.save(
            "./data/output.mp3", output.T, self._sampling_rate, channels_first=False
        )

        print(os.path.abspath("./data/output.mp3"))
        return os.path.abspath("./data/output.mp3")
