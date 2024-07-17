import argparse

import gradio as gr

from audioenhancer.constants import SAMPLING_RATE
from audioenhancer.inference import Inference

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    type=str,
    required=False,
    default="data/model/model_3700.pt",
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
    default=True,
)


args = parser.parse_args()

Inference = Inference(args.model_path, args.sampling_rate)


def enhancer(filepath):
    if not filepath:
        return None

    return Inference.inference(filepath)


article = """
Enhance the audio file. Please wait that the audio is uploaded before clicking on the Submit button.
"""

demo = gr.Interface(enhancer, gr.Audio(type="filepath"), "audio", article=article)

demo.launch(share=args.share)
