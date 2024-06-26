"""
Code for inference.
"""

import argparse

import torch
import torchaudio

from audioenhancer.constants import SAMPLING_RATE, MAX_AUDIO_LENGTH
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

model = SoundStream(
    D=512,
    C=64,
    strides=(2, 4, 4, 5),
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

    return waveform


audio = load(args.audio)

output = torch.Tensor()

for i in range(0, audio.size(2), int(SAMPLING_RATE * MAX_AUDIO_LENGTH)):
    chunk = audio[:, :, i : i + int(SAMPLING_RATE * MAX_AUDIO_LENGTH)]
    with torch.no_grad():
        output = torch.cat([output, model(chunk)], dim=2)

output = output.squeeze(0)

# fix runtime error: numpy
output = output.detach()
audio = audio.squeeze(0)

torchaudio.save("./data/input.mp3", audio.T, SAMPLING_RATE, channels_first=False)
torchaudio.save("./data/output.mp3", output.T, SAMPLING_RATE, channels_first=False)
