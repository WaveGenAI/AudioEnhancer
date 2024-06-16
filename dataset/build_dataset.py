"""
Module to build the dataset
"""

import os
import shutil

from dataset.codec import Encodec

CODEC = [Encodec()]


def create_dir(dir_path: str):
    """Create a directory if it does not exist.

    Args:
        dir_path (str): the directory path
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for used_codec in CODEC:
        if not os.path.exists(os.path.join(dir_path, str(used_codec))):
            os.makedirs(os.path.join(dir_path, str(used_codec)))


def build_ds(audio_dir: str, dataset_dir: str):
    """Build the dataset by converting audio files to low quality audio.

    Args:
        audio_dir (str): the directory containing the audio files
        dataset_dir (str): the directory to save the dataset
    """

    print("Building the dataset...")
    create_dir(dataset_dir)

    for files in os.listdir(audio_dir):
        for used_codec in CODEC:
            file_path = os.path.join(audio_dir, files)
            target_path = os.path.join(dataset_dir, str(used_codec), files)

            # copy the file to the dataset directory
            shutil.copy(file_path, dataset_dir)

            used_codec.encoder_decoder(file_path, target_path)
