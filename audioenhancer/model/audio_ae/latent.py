"""This module contains all the process for the latent space of the audio autoencoder."""

from torch import nn

from audioenhancer.model.audio_ae.mamba import MambaBlock


class LatentConfig:
    """
    Configuration for the latent space of the audio autoencoder.
    """

    def __init__(
            self,
            latent_dim,
            num_layer,
            state_size=64,
            conv_kernel=3,
            intermediate_size=2048,
            time_step_max=0.001,
            time_step_min=0.1,
            A_init_range=(0.001, 0.1),
            ssm_num_head=16,
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


class LatentProcessor(nn.Module):
    """
    This module processes the latent space of the audio autoencoder.
    """

    def __init__(self, in_dim: int, out_dim: int, num_code_book: int, latent_dim, num_layer):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layer = num_layer
        config = LatentConfig(latent_dim, num_layer)

        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)
        self.code_proj = nn.Linear(latent_dim, num_code_book)

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(MambaBlock(config))

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(x), self.code_proj(x)