"""
Code for inference.
"""

import os

import dac
import torch
import torchaudio
from einops import rearrange

from audioenhancer.model.audio_ae.model import model_xtransformer as model

import librosa
import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf


def remove_noise(audio_path, output_path):
    # Charger le fichier audio
    y, sr = librosa.load(audio_path)

    # Calculer le RMS (Root Mean Square) du signal original
    rms_original = np.sqrt(np.mean(y**2))

    # Définir les paramètres du filtre passe-bas
    cutoff = 7000  # Fréquence de coupure à 7000 Hz
    order = 2  # Ordre du filtre à 2
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist

    # Créer le filtre passe-bas
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Appliquer le filtre
    y_filtered = filtfilt(b, a, y)

    # Mélanger le signal original et le signal filtré
    y_output = y_filtered  # 85% original, 15% filtré

    # Normaliser le volume du signal de sortie
    rms_output = np.sqrt(np.mean(y_output**2))
    y_output = y_output * (rms_original / rms_output)

    # Sauvegarder le résultat
    sf.write(output_path, y_output, sr)


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
        self._autoencoder.eval()
        self._autoencoder.requires_grad_(False)

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

    def inference(self, audio_path: str, chunk_duration: int = 4):
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
                z_q, _, _, _, _ = self._autoencoder.quantizer(pred.float(), None)

                decoded = self._autoencoder.decode(z_q)

                decoded = decoded.transpose(0, 1)

                output = torch.cat([output, decoded], dim=2)

        # fix runtime error: numpy
        output = output.squeeze(0).detach().cpu()
        ae_input = ae_input.squeeze(0).detach().cpu()

        # cut the audio to the original size
        output = output[:, : audio.size(2)]

        torchaudio.save(
            "./data/input.mp3",
            ae_input.T,
            self._sampling_rate,
            channels_first=False,
        )
        torchaudio.save(
            "./data/output.mp3", output.T, self._sampling_rate, channels_first=False
        )

        # remove_noise("./data/output.mp3", "./data/output.mp3")

        return os.path.abspath("./data/output.mp3")
