"""
Code for inference.
"""

import argparse
import math

import torch
import torchaudio

from audioenhancer.constants import MAX_AUDIO_LENGTH, SAMPLING_RATE
from audioenhancer.model.audio_ae.model import model_xtransformer as model
from archisound import ArchiSound

from einops import rearrange
import torch.nn.functional as F

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

model.load_state_dict(torch.load(args.model_path))
CHUNCK_SIZE = 2 ** math.ceil(math.log2(MAX_AUDIO_LENGTH * args.sampling_rate))


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
model.eval()
autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")
autoencoder.to(device)

output = output.to(device)

model.eval()

output_encoder_only = torch.Tensor()
output_encoder_only = output_encoder_only.to(device)

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
        print(f"Processing chunk {i}", end="\r")
        encoded = autoencoder.encode(chunk) * 0.1

        decodec = autoencoder.decode(encoded / 0.1, num_steps=40)
        output_encoder_only = torch.cat([output_encoder_only, decodec], dim=2)

        encoded = rearrange(encoded, "b c t -> b t c")
        pred = model(encoded)

        pred = rearrange(pred, "b t c -> b c t")

        decodec = autoencoder.decode(pred / 0.1, num_steps=20)

        output = torch.cat([output, decodec], dim=2)

output = output.squeeze(0)

# fix runtime error: numpy
output = output.detach().cpu()
audio = audio.squeeze(0).detach().cpu()

output_encoder_only = output_encoder_only.squeeze(0).detach().cpu()


torchaudio.save(
    "./data/input.mp3", output_encoder_only.T, args.sampling_rate, channels_first=False
)
torchaudio.save("./data/output.mp3", output.T, args.sampling_rate, channels_first=False)
