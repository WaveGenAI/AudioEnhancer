"""
Code for inference.
"""

import argparse

import torch
import torchaudio

from audioenhancer.constants import SAMPLING_RATE, MAX_AUDIO_LENGTH
from audioenhancer.model.audio_ae.auto_encoder import AutoEncoder1d

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
    default="data/model/model_2000.pt",
    help="The path to the model",
)

args = parser.parse_args()

model = AutoEncoder1d(
    in_channels=2,  # Number of input channels
    channels=32,  # Number of base channels
    multipliers=[
        2,
        4,
        8,
        12,
        16,
    ],  # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
    factors=[2, 4, 4, 8,],  # Downsampling/upsampling factor per layer
    num_blocks=[2, 2, 2, 2,],  # Number of resnet blocks per layer
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


audio = load(args.audio).cuda()

output = torch.Tensor().cuda()
model.eval()
model.cuda()

for i in range(0, audio.size(2), int(SAMPLING_RATE * MAX_AUDIO_LENGTH)):
    chunk = audio[:, :, i : i + int(SAMPLING_RATE * MAX_AUDIO_LENGTH)]
    with torch.no_grad():
        output = torch.cat([output, model(chunk)], dim=2)

output = output.squeeze(0)

# fix runtime error: numpy
output = output.detach().cpu()
audio = audio.squeeze(0).detach().cpu()

torchaudio.save("./data/input.mp3", audio.T, SAMPLING_RATE, channels_first=False)
torchaudio.save("./data/output.mp3", output.T, SAMPLING_RATE, channels_first=False)
