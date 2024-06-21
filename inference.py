"""
Code for inference.
"""

import torch
import torchaudio

from model.soundstream import SoundStream

model = SoundStream(
    D=256,
    C=58,
    strides=(2, 4, 5, 5),
)

model.load_state_dict(torch.load("data/model.pth"))


def load(waveform_path):
    waveform, sample_rate = torchaudio.load(waveform_path)
    resampler = torchaudio.transforms.Resample(sample_rate, 16000, dtype=waveform.dtype)
    waveform = resampler(waveform)

    waveform = waveform.mean(dim=0, keepdim=True)

    return torch.unsqueeze(waveform, dim=0)


audio = load("data/test.mp3")
output = model(audio)

torchaudio.save("data/output.wav", output[0], 16000)
