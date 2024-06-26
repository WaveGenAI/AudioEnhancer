"""This module contains the wave discriminator model."""

import torch
from torch import nn

from audioenhancer.model.encoder import Encoder
from audioenhancer.model.latent import Latent


class Pooler(nn.Module):
    """
    Pooler layer for the discriminator. This class come from the bert pooler:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L736
    """

    def __init__(self, d_model: int = 256):
        """
        Pooler layer for the discriminator.
        Selects the first token of the sequence and applies a linear layer with a tanh activation.

        Args:
            d_model (int, optional): The dimension of the model. Defaults to 256.
        """
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Discriminator(nn.Module):
    """Wave discriminator model."""

    def __init__(self, latent_dim, num_channels, strides=(2, 4, 5, 8)):
        """
        Wave discriminator model.

        Args:
            latent_dim (int): The latent dimension
            num_channels (int): The number of channels
            strides (tuple, optional): The strides for the encoder's convolutional layers. Defaults to (2, 4, 5, 8).
        """
        super().__init__()
        self.encoder = Encoder(C=num_channels, D=latent_dim, strides=strides)
        self.latent = Latent(d_model=latent_dim, num_layers=4)
        self.pooler = Pooler(d_model=latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)
        self.init_weights()
        torch.cuda.empty_cache()

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
        e, _ = self.encoder(x)
        l = self.latent(e).permute(0, 2, 1)
        p = self.pooler(l)
        logits = self.classifier(p)
        return logits
