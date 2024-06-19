"""
This module contains the data loader script.
"""

from datasets import Audio, Dataset

from dataset.load_dataset import load_dataset


# if the clean audio duration is longer than the compressed audio duration,
# then clean audio is cut to match the compressed audio duration
def _cut_audio(row: dict) -> dict:
    """Function that cuts the audio.

    Args:
        row (dict): The row to cut.

    Returns:
        dict: The row with the audio cut.
    """
    
    SAMPLE_RATE = row["clean"]['sampling_rate']
    
    clean_audio = row["clean"]["array"]
    clean_duration = int(clean_audio.shape[0] / SAMPLE_RATE)

    compressed_audio = row["compressed"]["array"]
    compressed_duration = int(compressed_audio.shape[0] / SAMPLE_RATE)
    
    if clean_duration > compressed_duration:
        row["clean"]["array"] = clean_audio[: compressed_duration * SAMPLE_RATE]

    return row


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

    dataset = dataset.cast_column("clean", Audio(sampling_rate=sampling_rate)).cast_column(
        "compressed", Audio(sampling_rate=sampling_rate)
    )

    return dataset


def prepare_data(dataset: Dataset) -> Dataset:
    """Function that prepares the dataset.

    Args:
        dataset (Dataset): The dataset to prepare.

    Returns:
        Dataset: The prepared dataset.
    """

    dataset = dataset.map(_cut_audio, remove_columns=dataset.column_names)

    return dataset
