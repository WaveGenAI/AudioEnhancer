"""
This script loads the dataset from the given path.
"""
import glob
import os

from datasets import Dataset


def load_dataset(path: str) -> Dataset:
    """
    Function that gathers the dataset.
    """
    def gen():
        i = 4
        clean_audio = glob.glob(path + "/*.mp3")
        codecs = [f for f in glob.glob(path+"/*", ) if os.path.isdir(f)]
        for i in range(len(clean_audio)):
            return_dict = {"clean": os.path.abspath(clean_audio[i])}
            for codec in codecs:
                return_dict = {
                    "clean": os.path.abspath(clean_audio[i]),
                    "compressed": os.path.abspath(glob.glob(str(codec) + "/*.mp3")[i]),
                    "type": codec.split('/')[-1]
                
                }
                yield return_dict

    return Dataset.from_generator(gen)
