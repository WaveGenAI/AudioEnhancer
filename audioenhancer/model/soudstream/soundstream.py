""" 
Module to build the model
Note: All the code come from https://github.com/haydenshively/SoundStream/tree/master
"""

from functools import reduce

import torch
from torch import nn

from audioenhancer.model.soudstream.decoder import Decoder
from audioenhancer.model.soudstream.encoder import Encoder
from audioenhancer.model.soudstream.latent import Latent


class SoundStream(nn.Module):
    """SoundStream model."""

    def __init__(self, D, C, strides=(2, 4, 5, 8), residual: bool = False):
        """
        SoundStream model.

        Args:
            D (int): The latent space dimension
            C (int): The number of channels
            strides (tuple, optional): The strides for the encoder's convolutional layers. Defaults to (2, 4, 5, 8).
        """
        super().__init__()
        # The temporal resampling ratio between input waveform and embeddings.
        # Not used in here, but helpful for consumers.
        self.M = reduce(lambda a, b: a * b, strides)

        self.encoder = Encoder(C=C, D=D, strides=strides, residual=residual)
        self.decoder = Decoder(C=C, D=D, strides=strides, residual=residual)
        self.latent = Latent(d_model=D)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights with a normal distribution.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x: batch_size x 1 x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x 1 x (T / 1)

        e, skips_conn = self.encoder(x)

        e = self.latent(e)
        o = self.decoder(e, skips_conn)

        return o
