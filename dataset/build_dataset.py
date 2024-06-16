"""
Module to build the dataset
"""

import os

from dataset.codec import *

CODEC = [Encodec()]


def create_dir(dir_path: str):
    """Create a directory if it does not exist.

    Args:
        dir_path (str): the directory path
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for codec in CODEC:
        os.makedirs(os.path.join(dir_path, str(codec)))


def build_ds(audio_dir: str, dataset_dir: str):
    """Build the dataset by converting audio files to low quality audio.

    Args:
        audio_dir (str): the directory containing the audio files
        dataset_dir (str): the directory to save the dataset
    """

    print("Building the dataset...")
    create_dir(dataset_dir)

    for files in os.listdir(audio_dir):
        for codec in CODEC:
            file_path = os.path.join(audio_dir, files)
            encoded = codec.encode(file_path)
            decodec = codec.decode(encoded)
