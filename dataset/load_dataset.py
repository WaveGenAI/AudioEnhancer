"""
This script loads the dataset from the given path.
"""

import glob
import os

from datasets import Audio, Dataset, Features


def load_dataset(path: str, sampling_rate: int) -> Dataset:
    """Function that loads the dataset.

    Args:
        path (str): The path to the dataset.
        sampling_rate (int): The sampling rate of the audio.

    Returns:
        Dataset: The dataset loaded.
    """

    def gen():
        i = 4
        clean_audio = glob.glob(path + "/*.mp3")
        codecs = [
            f
            for f in glob.glob(
                path + "/*",
            )
            if os.path.isdir(f)
        ]
        for i in range(len(clean_audio)):
            for codec in codecs:
                try:
                    return_dict = {
                        "clean": os.path.abspath(clean_audio[i]),
                        "compressed": os.path.abspath(
                            os.path.join(codec, os.path.basename(clean_audio[i]))
                        ),
                    }
                    yield return_dict
                except IndexError:
                    pass

    features = Features(
        {
            "clean": Audio(sampling_rate=sampling_rate),
            "compressed": Audio(sampling_rate=sampling_rate),
        }
    )
    return Dataset.from_generator(gen, features=features)
