"""
Latent layer for the model
"""

import torch
from torch import nn


class Latent(nn.Module):
    """Latent layer for the model"""
    def __init__(self, d_model: int = 256, intermediate_dim: int = 1024, num_layers: int = 4):
        """
        Latent layer for the model

        This layer is composed of a TransformerEncoder layer that will process the latent space.

        Args:
            d_model (int, optional): The dimension of the model. Defaults to 256.
            intermediate_dim (int, optional): The intermediate dimension. Defaults to 1024.
            num_layers (int, optional): The number of layers. Defaults to 4.
        """
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
