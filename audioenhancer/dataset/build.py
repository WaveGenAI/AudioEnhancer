"""
Module to build the dataset
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List

import tqdm
from pydub import AudioSegment

from audioenhancer.dataset.codec import Codec


class DatasetBuilder:
    """
    Class to generate the dataset.
    """

    def __init__(self, codec: List[Codec]):
        self.codec = codec
        self._nbm_files_processed = 0

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

    def process_file(
        self,
        audio_dir: str,
        filename: str,
        target_dir: str,
        max_duration_ms: int = 10000,
    ):
        """Process a file by splitting it into segments.

        Args:
            audio_dir (str): the directory containing the audio files
            filename (str): the name of the file,
            target_dir (str): the directory to save the segments
            max_duration_ms (int, optional): the max duration of the audio. Defaults to 10000.
        """
        audio_path = os.path.join(audio_dir, filename)
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio)

        self._nbm_files_processed += 1
        print(f"Process {self._nbm_files_processed} files", end="\r")

        if total_duration <= (max_duration_ms + 0.2):
            return

        for i in range(0, total_duration, max_duration_ms):
            segment = audio[i : i + max_duration_ms]

            # pad the segment if it is shorter than the max duration
            if len(segment) < max_duration_ms:
                segment = segment + AudioSegment.silent(
                    duration=max_duration_ms - len(segment)
                )

            segment_filename = (
                f"{os.path.splitext(filename)[0]}_part{i // max_duration_ms + 1}.mp3"
            )
            output_path = os.path.join(target_dir, segment_filename)

            segment.export(output_path, format="mp3")

    def process_audio_files(
        self, audio_dir: str, target_dir: str, max_duration_ms: int = 10000
    ):
        """
        Process audio files by splitting them into segments.

        Args:
            audio_dir (str): the directory containing the audio files
            target_dir (str): the directory to save the segments
            max_duration_ms (int, optional): the maximal duration in ms of the audio. Defaults to 10000.
        """

        filename_list = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.process_file, audio_dir, filename, target_dir, max_duration_ms
                )
                for filename in filename_list
            ]
            for future in futures:
                future.result()

        print("All files processed")

    def build_ds(
        self,
        audio_dir: str,
        dataset_dir: str,
        split_audio: bool = False,
        max_duration_ms: int = 10000,
    ):
        """Build the dataset by converting audio files to low quality audio.

        Args:
            audio_dir (str): the directory containing the audio files
            dataset_dir (str): the directory to save the dataset
            split_audio (bool, optional): whether to split the audio files into segments. Defaults to False.
            max_duration_ms (int, optional): the maximal duration in ms of the audio. Defaults to 10000.
        """

        print("Building the dataset...")

        self.create_dir(dataset_dir)

        if split_audio:
            self.process_audio_files(audio_dir, dataset_dir, max_duration_ms)

        total_files = len(os.listdir(dataset_dir))
        total_operations = total_files * len(self.codec)
        progress_bar = tqdm.tqdm(
            total=total_operations, desc="Processing files", unit="file"
        )

        for used_codec in self.codec:
            for files in os.listdir(dataset_dir):
                if not files.endswith(".mp3"):
                    continue

                file_path = os.path.join(dataset_dir, files)
                target_path = os.path.join(dataset_dir, str(used_codec), files)

                used_codec.encoder_decoder(file_path, target_path)

                progress_bar.update(1)
