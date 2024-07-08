"""
Code for inference.
"""

import argparse

from audioenhancer.constants import SAMPLING_RATE
from audioenhancer.inference import Inference

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio",
    default="../media/works/dataset/opus/5700_part2.mp3",  # ../media/works/dataset/dac/5700_part2.mp3
    type=str,
    required=False,
    help="The path to the audio file to enhance",
)

parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="data/model/model_300.pt",
    help="The path to the model",
)

parser.add_argument(
    "--sampling_rate",
    type=int,
    required=False,
    default=SAMPLING_RATE,
    help="The sampling rate of the audio",
)

args = parser.parse_args()

inference = Inference(args.model_path, args.sampling_rate)

inference.inference(args.audio)
