"""
This module contains the data loader script.
"""

from datasets import Audio, Dataset

from dataset.load_dataset import load_dataset


def load_data(path: str, sampling_rate: int = 16000) -> Dataset:
    """Function that loads the dataset.

    Args:
        path (str): The path to the dataset.
        sampling_rate (int): The sampling rate of the audio.

    Returns:
        Dataset: The dataset loaded.
    """

    dataset = load_dataset(path)
    dataset = dataset.shuffle(seed=42)

    dataset = dataset.cast_column(
        "clean", Audio(sampling_rate=sampling_rate)
    ).cast_column("compressed", Audio(sampling_rate=sampling_rate))

    return dataset
