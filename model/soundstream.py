""" 
Module to build the model
Note: All the code come from https://github.com/haydenshively/SoundStream/tree/master
"""

from functools import reduce

import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder


class SoundStream(nn.Module):
    def __init__(self, D, C, strides=(2, 4, 5, 8)):
        super(SoundStream, self).__init__()

        # The temporal resampling ratio between input waveform and embeddings.
        # Not used in here, but helpful for consumers.
        self.M = reduce(lambda a, b: a * b, strides)

        self.encoder = Encoder(C=C, D=D, strides=strides)
        self.decoder = Decoder(C=C, D=D, strides=strides)

    def forward(self, x):
        # x: batch_size x 1 x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x 1 x (T / 1)
    
        e = self.encoder(x)
        o = self.decoder(e)

        return o
