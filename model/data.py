"""
This module contains the data loader script.
"""

import torch
from datasets import Dataset

from constants import MAX_AUDIO_LENGTH, SAMPLING_RATE
from dataset.load_dataset import load_dataset

MAX_ARRAY_LENGTH = MAX_AUDIO_LENGTH * SAMPLING_RATE


def prepare_dataset(batch, max_array_length):
    """Function that prepares the dataset.

    Args:
        batch (dict): The batch.
    """

    if batch["clean"]["array"].shape[-1] < max_array_length:
        batch["clean"]["array"] = torch.nn.functional.pad(
            batch["clean"]["array"],
            (0, max_array_length - batch["clean"]["array"].shape[-1]),
            "constant",
            0,
        )

    if batch["compressed"]["array"].shape[-1] < max_array_length:
        batch["compressed"]["array"] = torch.nn.functional.pad(
            batch["compressed"]["array"],
            (0, max_array_length - batch["compressed"]["array"].shape[-1]),
            "constant",
            0,
        )

    return batch


def load_data(path: str, sampling_rate: int = 16000) -> Dataset:
    """Function that loads the dataset.

    Args:
        path (str): The path to the dataset.
        sampling_rate (int): The sampling rate of the audio.

    Returns:
        Dataset: The dataset loaded.
    """

    dataset = load_dataset(path, sampling_rate)
    dataset = dataset.with_format("torch")
    dataset = dataset.shuffle(seed=42)

    dataset = dataset.map(
        lambda x: prepare_dataset(x, MAX_ARRAY_LENGTH),
    )

    return dataset
