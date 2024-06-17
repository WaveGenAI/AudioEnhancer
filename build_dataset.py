"""
This script is used to build the dataset from the audio files. 
The audio files are stored in the audio_dir and the dataset is stored in the dataset_dir. 
The codec is used to encode the audio files.
"""

import argparse

from dataset.build import DatasetBuilder
from dataset.codec import DAC, Encodec, Opus, Soundstream

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_dir",
    type=str,
    required=True,
    help="The directory containing the audio files",
)
parser.add_argument(
    "--dataset_dir", type=str, required=True, help="The directory to save the dataset"
)

parser.add_argument(
    "--codec",
    type=str,
    nargs="+",
    required=True,
    help="The codecs to use, supported codecs: dac, encodec, soundstream, opus. Example: --codec dac encodec",
)

args = parser.parse_args()

codec = []
for c in args.codec:
    if c == "dac":
        codec.append(DAC())
    elif c == "encodec":
        codec.append(Encodec())
    elif c == "soundstream":
        codec.append(Soundstream())
    elif c == "opus":
        codec.append(Opus())
    else:
        raise ValueError(f"Unknown codec: {c}")

builder = DatasetBuilder(codec)
builder.build_ds(args.audio_dir, args.dataset_dir)
