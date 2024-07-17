import warnings

import torch
from torch import nn
from x_transformers import XTransformer
from xformers.factory import xFormerConfig, xFormer


class Transformer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, latent_dim, num_layer):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.num_layer = num_layer

        transformer = XTransformer(
            dim=latent_dim,
            enc_depth=num_layer,
            enc_heads=16,
            enc_max_seq_len=0,
            enc_attn_flash=True,
            enc_num_tokens=256,
            enc_cross_attend=False,
            enc_ff_glu=True,
            enc_rotary_pos_emb=True,
            enc_use_scalenorm=True,
            enc_zero_init_branch_output=True,
            dec_num_tokens=256,
            dec_depth=num_layer,
            dec_heads=16,
            dec_ff_glu=True,
            dec_rotary_pos_emb=True,
            dec_use_scalenorm=True,
            dec_attn_flash=True,
            dec_max_seq_len=0,
            dec_zero_init_branch_output=True,
        )
        self.embed = nn.Parameter(torch.randn(latent_dim))
        self.encoders = transformer.encoder.attn_layers
        self.decoders = transformer.decoder.net.attn_layers
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)

        x = self.encoders(x)

        h = self.embed.expand(x.shape)
        h = self.decoders(h, context=x)
        h = self.out_proj(h)
        return x - h
