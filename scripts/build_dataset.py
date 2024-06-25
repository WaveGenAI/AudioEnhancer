"""
This script is used to build the dataset from the audio files. 
The audio files are stored in the audio_dir and the dataset is stored in the dataset_dir. 
The codec is used to encode the audio files.
"""

import argparse

from audioenhancer.dataset.build import DatasetBuilder
from audioenhancer.dataset.codec import DAC, Encodec, Opus, Soundstream

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_dir",
    type=str,
    default="./media/works/data/",
    required=False,
    help="The directory containing the audio files",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./media/works/dataset",
    required=False,
    help="The directory to save the dataset",
)

parser.add_argument(
    "--codec",
    type=str,
    nargs="+",
    default=["soundstream"],
    required=False,
    help="The codecs to use, supported codecs: dac, encodec, soundstream, opus. Example: --codec dac encodec",
)
parser.add_argument(
    "--split_audio",
    action="store_true",
    help="Split the audio files into segments",
)

parser.add_argument(
    "--max_duration_ms",
    type=int,
    default=10000,
    required=False,
    help="The maximal duration in ms of the audio",
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
builder.build_ds(
    args.audio_dir, args.dataset_dir, args.split_audio, args.max_duration_ms
)
