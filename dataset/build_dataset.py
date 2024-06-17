"""
Module to build the dataset
"""

import os
import shutil
from typing import List

import tqdm

from dataset.codec import Codec


class DatasetBuilder:
    """
    Class to generate the dataset.
    """

    def __init__(self, codec: List[Codec]):
        self.codec = codec

    def create_dir(self, dir_path: str):
        """Create a directory if it does not exist.

        Args:
            dir_path (str): the directory path
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for used_codec in self.codec:
            if not os.path.exists(os.path.join(dir_path, str(used_codec))):
                os.makedirs(os.path.join(dir_path, str(used_codec)))

    def build_ds(self, audio_dir: str, dataset_dir: str):
        """Build the dataset by converting audio files to low quality audio.

        Args:
            audio_dir (str): the directory containing the audio files
            dataset_dir (str): the directory to save the dataset
        """

        print("Building the dataset...")
        self.create_dir(dataset_dir)

        total_files = len(os.listdir(audio_dir))
        total_operations = total_files * len(self.codec)
        progress_bar = tqdm.tqdm(
            total=total_operations, desc="Processing files", unit="file"
        )

        for used_codec in self.codec:
            for files in os.listdir(audio_dir):
                file_path = os.path.join(audio_dir, files)

                # copy the file to the dataset directory
                shutil.copy(file_path, dataset_dir)

                target_path = os.path.join(dataset_dir, str(used_codec), files)
                used_codec.encoder_decoder(file_path, target_path)

                progress_bar.update(1)
