"""
Code for inference.
"""

import torch
import torchaudio

from model.soundstream import SoundStream
from constants import SAMPLING_RATE


model = SoundStream(
    D=256,
    C=58,
    strides=(2, 4, 5, 5),
)

# model.load_state_dict(torch.load("data/model.pth"))


def load(waveform_path):
    waveform, sample_rate = torchaudio.load(waveform_path)
    resampler = torchaudio.transforms.Resample(sample_rate, SAMPLING_RATE, dtype=waveform.dtype)
    waveform = resampler(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    return waveform

audio = load("/media/works/dataset/soundstream/12100_part13.mp3")

output = model(audio)

# torchaudio.save("data/output.wav", output, SAMPLING_RATE)
