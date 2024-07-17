"""This module contains all the process for the latent space of the audio autoencoder."""
import random

import torch
from torch import nn
from torch.nn import functional as F

from audioenhancer.model.audio_ae.expert import Expert
from audioenhancer.model.audio_ae.mamba import MambaBlock


class LatentConfig:
    """
    Configuration for the latent space of the audio autoencoder.
    """

    def __init__(
            self,
            latent_dim,
            num_layer,
            state_size=128,
            conv_kernel=3,
            intermediate_size=3072,
            time_step_max=0.001,
            time_step_min=0.1,
            A_init_range=(0.001, 0.1),
            ssm_num_head=24,
            use_conv_bias=True,
            use_bias=False,
            hidden_act="silu",
            residual_in_fp32=True,

    ):
        self.hidden_size = latent_dim
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.intermediate_size = intermediate_size
        self.time_step_max = time_step_max
        self.time_step_min = time_step_min
        self.A_init_range = A_init_range
        self.ssm_num_head = ssm_num_head
        self.use_conv_bias = use_conv_bias
        self.use_bias = use_bias
        self.hidden_act = hidden_act
        self.num_layer = num_layer
        self.residual_in_fp32 = residual_in_fp32


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


class LatentProcessor(nn.Module):
    """
    This module processes the latent space of the audio autoencoder.
    """

    def __init__(self, in_dim: int, out_dim: int, latent_dim, num_layer, noise_grad=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layer = num_layer
        config = LatentConfig(latent_dim, num_layer)

        self.in_proj = nn.Linear(in_dim, latent_dim)

        self.out_proj = nn.Linear(latent_dim, out_dim)

        self.mambas = nn.ModuleList([MambaBlock(config) for _ in range(num_layer)])
        self.unknow_noise = nn.Parameter(torch.randn(latent_dim))
        self.noise_embed = nn.Embedding(noise_grad, latent_dim)
        self.noise_head = nn.Linear(latent_dim, noise_grad)
        # self.pre_process = nn.Sequential(
        #     MambaBlock(config),
        #     MambaBlock(config),
        #     MambaBlock(config),
        #     MambaBlock(config),
        # )
        # self.classifier = nn.Sequential(
        #     Pooler(d_model=latent_dim),
        #     nn.Linear(latent_dim, num_expert),
        # )

    def classify(self, x):
        x = self.in_proj(x)
        x = self.pre_process(x)
        return self.classifier(x)

    def forward(self, x, noise, gen_noise=False, noise_label=None):
        bzs = x.size(0)
        h = self.in_proj(x)
        if noise is not None and not gen_noise:
            noise = self.noise_embed(noise).reshape(bzs, 1, -1)
            h = torch.cat([h, noise], dim=1)
            gen_noise = True
        else:
            noise = self.unknow_noise.reshape(1, 1, -1).repeat(bzs, 1, 1)
            h = torch.cat([h, noise], dim=1)

        # h = self.pre_process(h)
        for mamba in self.mambas:
            h = mamba(h, gen_noise=gen_noise)
        # if classes is not None:
        #     return x * classes[:, None, None, 0] + self.out_proj(h) * classes[:, None, None, 1]

        logits = self.noise_head(h[:, -1])
        h = h[:, :-1]
        if noise_label is not None:
            if not gen_noise:
                return self.out_proj(h), 0
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), noise_label.view(-1))
            return self.out_proj(h), loss

        return self.out_proj(h), logits

    # expert
    # def forward(self, x, expert_id=None):
    #     x = self.in_proj(x)
    #     if expert_id is None:
    #         x = self.pre_process(x)
    #         class_logits = self.classifier(x)
    #         weight = torch.softmax(class_logits, dim=-1)
    #         h = torch.sum((weight * self.expert(x).permute(2,3,1,0)).permute(3,2,0,1), dim=0)
    #         return self.out_proj(h), class_logits
    #     else:
    #         x = self.pre_process(x)
    #         class_logits = self.classifier(x)
    #         random_tensor = torch.randint(0, self.num_expert-1, expert_id.size()).to(expert_id.device)
    #
    #         # Replace 0 values in `tensor` with corresponding values from `random_tensor`
    #         expert_id = torch.where(expert_id == 0, random_tensor, expert_id)
    #         return self.out_proj(self.expert(x, expert_id)), class_logits