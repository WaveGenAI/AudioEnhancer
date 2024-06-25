"""
Latent layer for the model
"""

import torch
import torch.nn as nn


class Latent(nn.Module):
    def __init__(self, d_model: int = 256, intermediate_dim: int = 1024, num_layers: int = 4):
        super(Latent, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=16, dim_feedforward=intermediate_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """

        x = x.permute(0, 2, 1)

        x = self.transformer_encoder(x)

        x = x.permute(0, 2, 1)

        return x
