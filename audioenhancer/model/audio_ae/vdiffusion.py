import torch
import torch.nn.functional as F
import torch.nn as nn
from audio_diffusion_pytorch import VDiffusion
from audio_diffusion_pytorch.diffusion import extend_dim
from torch import Tensor
from typing import Sequence
from archisound import ArchiSound

from auraloss.freq import MultiResolutionSTFTLoss


class CustomVDiffusion(VDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)

        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * y
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)

        return self.loss_fn(v_pred, v_target)
