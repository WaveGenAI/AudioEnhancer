"""
This script is used to build the dataset from the audio files. 
The audio files are stored in the audio_dir and the dataset is stored in the dataset_dir. 
The codec is used to encode the audio files.
"""

import argparse

import setup_paths

from audioenhancer.dataset.download import download

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_dir",
    type=str,
    required=True,
    help="The directory where the audio files will be stored",
)
parser.add_argument(
    "--quantity",
    type=int,
    required=False,
    default=1000,
    help="The number of tar files to download from jamendo",
)

args = parser.parse_args()

download(args.audio_dir, args.quantity)
