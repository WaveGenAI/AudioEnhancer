import argparse

import gradio as gr

from audioenhancer.constants import SAMPLING_RATE
from audioenhancer.inference import Inference

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="data/model/model_2000.pt",
    help="The path to the model",
)

parser.add_argument(
    "--sampling_rate",
    type=int,
    required=False,
    default=SAMPLING_RATE,
    help="The sampling rate of the audio",
)

parser.add_argument(
    "--share",
    action="store_true",
    help="Share the interface",
    default=False,
)


args = parser.parse_args()

Inference = Inference(args.model_path, args.sampling_rate)


def enhancer(filepath):
    return Inference.inference(filepath)


demo = gr.Interface(enhancer, gr.Audio(type="filepath"), "audio")

demo.launch(share=args.share)
