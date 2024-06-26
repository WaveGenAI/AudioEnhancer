"""
Code for inference.
"""

import argparse

import torch
import torchaudio
from audio_encoders_pytorch import AutoEncoder1d

from audioenhancer.constants import SAMPLING_RATE
from audioenhancer.model.soundstream import SoundStream

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio",
    default="../data/test.mp3",
    type=str,
    required=False,
    help="The path to the audio file to enhance",
)

parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="data/model/model_6500.pt",
    help="The path to the model",
)

args = parser.parse_args()

model = AutoEncoder1d(
    in_channels=2,  # Number of input channels
    channels=32,  # Number of base channels
    multipliers=[
        1,
        1,
        2,
        2,
    ],  # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
    factors=[4, 4, 4],  # Downsampling/upsampling factor per layer
    num_blocks=[2, 2, 2],  # Number of resnet blocks per layer
)

model.load_state_dict(torch.load(args.model_path))


def load(waveform_path):
    """
    Load the waveform from the given path and resample it to the desired sampling rate.

    Args:
        waveform_path (str): The path to the waveform file.
    """
    waveform, sample_rate = torchaudio.load(waveform_path)
    resampler = torchaudio.transforms.Resample(
        sample_rate, SAMPLING_RATE, dtype=waveform.dtype
    )
    waveform = resampler(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    waveform = waveform.unsqueeze(0)

    return waveform[:, :, : SAMPLING_RATE * 10]


audio = load(args.audio)

output = model(audio)
output = output.squeeze(0)

# fix runtime error: numpy
output = output.detach()
audio = audio.squeeze(0)

torchaudio.save("./data/input.mp3", audio.T, SAMPLING_RATE, channels_first=False)
torchaudio.save("./data/output.mp3", output.T, SAMPLING_RATE, channels_first=False)
