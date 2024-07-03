"""
Code for inference.
"""

import argparse
import math

import dac
import torch
import torchaudio
from dac import DACFile
from einops import rearrange

from audioenhancer.constants import MAX_AUDIO_LENGTH, SAMPLING_RATE
from audioenhancer.model.audio_ae.model import mamba_model as model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio",
    default="../media/works/dataset/opus/5700_part2.mp3",
    type=str,
    required=False,
    help="The path to the audio file to enhance",
)

parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="data/model/model_200.pt",
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


CHUNCK_SIZE = 2 ** math.ceil(math.log2(MAX_AUDIO_LENGTH * args.sampling_rate))
model.load_state_dict(torch.load(args.model_path))

autoencoder_path = dac.utils.download(model_type="44khz")
autoencoder = dac.DAC.load(autoencoder_path).to("cuda")
audio = load(args.audio)
output = torch.Tensor()
ae_input = torch.Tensor()

device = torch.device("cuda")
audio = audio.to(device)
model = model.to(device)
autoencoder.to(device)
output = output.to(device)
ae_input = ae_input.to(device)

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

        z, orig_code, latent, _, _ = autoencoder.encode(chunk.transpose(0, 1))

        # create input for the model
        z_q = 0
        latent_split = torch.split(latent, 8, dim=1)
        for i, quantizer in enumerate(autoencoder.quantizer.quantizers):
            z_q_i, _ = quantizer.decode_latents(latent_split[i])
            z_q_i = (
                    latent_split[i] + (z_q_i - latent_split[i]).detach()
            )
            z_q_i = quantizer.out_proj(z_q_i)
            mask = (
                    torch.full((z_q_i.shape[0],), fill_value=i, device=z_q_i.device) < 9
            )
            z_q = z_q + z_q_i * mask[:, None, None]

        decoded = autoencoder.decode(z_q)

        decoded = decoded.transpose(0, 1)

        ae_input = torch.cat([ae_input, decoded], dim=2)

        # create output for the model

        encoded = latent.unsqueeze(0)
        c, d = encoded.shape[1], encoded.shape[2]
        encoded = rearrange(encoded, "b c d t -> b (t c) d")

        # predict codebook
        pred, codes = model(encoded)
        # codes = torch.argmax(codes, dim=-1)
        # codes = rearrange(codes, " (b t c) d ->b c d t", b=1, c=2, d=9)
        #
        # codes = codes.squeeze(0).to(torch.int32)
        # z = autoencoder.quantizer.from_codes(codes)[0]

        z = rearrange(pred, "b (t c) d -> b c d t", c=c, d=d).squeeze(0)
        z_q = 0
        latent = torch.split(latent, 8, dim=1)
        for i, quantizer in enumerate(autoencoder.quantizer.quantizers):
            z_q_i, _ = quantizer.decode_latents(latent[i])
            z_q_i = (
                    latent[i] + (z_q_i - latent[i]).detach()
            )
            z_q_i = quantizer.out_proj(z_q_i)
            mask = (
                    torch.full((z_q_i.shape[0],), fill_value=i, device=z_q_i.device) < 9
            )
            z_q = z_q + z_q_i * mask[:, None, None]

        decoded = autoencoder.decode(z_q)

        decoded = decoded.transpose(0, 1)

        output = torch.cat([output, decoded], dim=2)

# fix runtime error: numpy
output = output.squeeze(0).detach().cpu()
ae_input = ae_input.squeeze(0).detach().cpu()


torchaudio.save(
    "./data/input.mp3",
    ae_input.T,
    args.sampling_rate,
    channels_first=False,
)
torchaudio.save("./data/output.mp3", output.T, args.sampling_rate, channels_first=False)
