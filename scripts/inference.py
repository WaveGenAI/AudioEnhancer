"""
Code for inference.
"""

import argparse

import torch
import torchaudio

from audioenhancer.constants import SAMPLING_RATE
from audioenhancer.model.soundstream import SoundStream

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio",
    type=str,
    required=True,
    help="The path to the audio file to enhance",
)

args = parser.parse_args()

model = SoundStream(
    D=256,
    C=58,
    strides=(2, 4, 5, 5),
)

model.load_state_dict(torch.load("data/model.pth"))


def load(waveform_path):
    waveform, sample_rate = torchaudio.load(waveform_path)
    resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE, dtype=waveform.dtype)
    waveform = resampler(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
  
    waveform = waveform.unsqueeze(0)
    
    return waveform[:,:, :SAMPLING_RATE*10]

audio = load(args.audio)

output = model(audio)
output = output.squeeze(0)

torchaudio.save("data/output.wav", output, SAMPLING_RATE)