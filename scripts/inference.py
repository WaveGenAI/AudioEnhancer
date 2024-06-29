"""
Code for inference.
"""

import argparse

import torch
import torchaudio
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VSampler

from audioenhancer.constants import MAX_AUDIO_LENGTH, SAMPLING_RATE
from audioenhancer.model.audio_ae.vdiffusion import CustomVDiffusion

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

parser.add_argument(
    "--sampling_rate",
    type=int,
    required=False,
    default=SAMPLING_RATE,
    help="The sampling rate of the audio",
)

args = parser.parse_args()


model = DiffusionModel(
    net_t=UNetV0,
    in_channels=2,  # U-Net: number of input channels
    channels=[256, 512, 1024, 1024, 1024, 1024],  # U-Net: channels at each layer
    factors=[
        4,
        4,
        4,
        4,
        4,
        4,
    ],  # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 2, 2, 2],  # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=8,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=CustomVDiffusion,  # The diffusion method used
    sampler_t=VSampler,  # The diffusion sampler used
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
        sample_rate, args.sampling_rate, dtype=waveform.dtype
    )
    waveform = resampler(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    waveform = waveform.unsqueeze(0)

    return waveform


device = torch.device("cuda")

audio = load(args.audio)
output = torch.Tensor()

audio = audio.to(device)
model = model.to(device)
output = output.to(device)

model.eval()

CHUNCK_SIZE = 2**18
for i in range(0, audio.size(2), int(CHUNCK_SIZE)):
    chunk = audio[:, :, i : i + int(CHUNCK_SIZE)]

    if chunk.size(2) < int(CHUNCK_SIZE):
        chunk = torch.nn.functional.pad(
            chunk,
            (0, int(CHUNCK_SIZE) - chunk.size(2)),
            "constant",
            0,
        )

    with torch.no_grad():
        pred = model.sample(chunk, num_steps=20)
        output = torch.cat([output, pred], dim=2)

output = output.squeeze(0)

# fix runtime error: numpy
output = output.detach().cpu()
audio = audio.squeeze(0).detach().cpu()

torchaudio.save("./data/input.mp3", audio.T, args.sampling_rate, channels_first=False)
torchaudio.save("./data/output.mp3", output.T, args.sampling_rate, channels_first=False)
