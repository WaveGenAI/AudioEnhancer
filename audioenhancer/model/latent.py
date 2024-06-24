"""
Latent layer for the model
"""

import torch
import torch.nn as nn


class Latent(nn.Module):
    def __init__(self, D: int = 256):
        super(Latent, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=16, dim_feedforward=1024
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """

        x = x.permute(2, 0, 1)

        x = self.transformer_encoder(x)

        x = x.permute(1, 2, 0)

        return x
