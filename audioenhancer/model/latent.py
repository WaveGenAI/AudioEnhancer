"""
Latent layer for the model
"""

import torch
from torch import nn


class Latent(nn.Module):
    """Latent layer for the model"""

    def __init__(self, d_model: int = 256, num_layers: int = 4):
        """
        Latent layer for the model

        This layer is composed of a a list of GRU layer that will process the latent space.

        Args:
            d_model (int, optional): The dimension of the model. Defaults to 256.
            num_layers (int, optional): The number of layers. Defaults to 4.
        """
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = x.permute(0, 2, 1)

        x, _ = self.gru(x)

        x = x.permute(0, 2, 1)

        x = self.relu(x)
        x = self.dropout(x)

        return x
