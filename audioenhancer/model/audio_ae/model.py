"""
Code for the model.
"""

import torch
from archisound import ArchiSound
from audio_diffusion_pytorch import (
    DiffusionAE,
    DiffusionModel,
    UNetV0,
    VDiffusion,
    VSampler,
)
from audio_encoders_pytorch import AutoEncoder1d, MelE1d, TanhBottleneck
from auraloss.freq import MultiResolutionSTFTLoss
from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder

from audioenhancer.model.audio_ae.latent import LatentProcessor
from audioenhancer.model.audio_ae.vdiffusion import CustomVDiffusion

model = DiffusionModel(
    net_t=UNetV0,
    in_channels=2,
    channels=[128, 256, 256, 256, 512, 512, 512, 768, 768],
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
    attention_heads=12,
    attention_features=64,
    diffusion_t=CustomVDiffusion,
    sampler_t=VSampler,
)

model_v1 = DiffusionModel(
    net_t=UNetV0,
    in_channels=32,  # U-Net: number of input/output (audio) channels
    channels=[128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
    factors=[
        1,
        2,
        2,
        2,
        2,
        2,
    ],  # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 4, 8, 8],  # U-Net: number of repeating items at each layer
    attentions=[0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=12,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=CustomVDiffusion,
    sampler_t=VSampler,
)

model_mediumV2 = DiffusionModel(
    net_t=UNetV0,
    in_channels=32,
    channels=[256, 256, 256, 256, 512, 512, 512, 768, 768],
    factors=[1, 1, 1, 1, 2, 2, 2, 2, 2],
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
    attention_heads=12,
    attention_features=64,
    diffusion_t=CustomVDiffusion,
    sampler_t=VSampler,
)

model_medium = DiffusionModel(
    net_t=UNetV0,
    in_channels=32,  # U-Net: number of input/output (audio) channels
    channels=[128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
    factors=[
        1,
        2,
        2,
        2,
        2,
        2,
    ],  # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 4, 8, 8],  # U-Net: number of repeating items at each layer
    attentions=[0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=12,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion,
    sampler_t=VSampler,
)

model_medium_compr = DiffusionModel(
    net_t=UNetV0,
    dim=1,
    in_channels=32,
    channels=[32, 32, 64, 64, 128, 128, 256, 256],
    factors=[2, 2, 2, 2, 2, 2, 2, 2],
    items=[2, 2, 2, 2, 2, 2, 4, 4],
    attentions=[0, 0, 0, 0, 0, 1, 1, 1],
    attention_heads=8,
    attention_features=64,
    diffusion_t=CustomVDiffusion,
    sampler_t=VSampler,
)

model_diffusion_ainur = DiffusionModel(
    net_t=UNetV0,
    in_channels=32,  # U-Net: number of input/output (audio) channels
    channels=[128, 256, 512, 512, 1024, 1024],  # U-Net: channels at each layer
    factors=[
        1,
        2,
        2,
        2,
        2,
        2,
    ],  # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 4, 8, 8],  # U-Net: number of repeating items at each layer
    attentions=[0, 0, 1, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=12,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=CustomVDiffusion,
    sampler_t=VSampler,
)

model_ae = DiffusionAE(
    encoder=MelE1d(  # The encoder used, in this case a mel-spectrogram encoder
        in_channels=2,
        channels=512,
        multipliers=[1, 1],
        factors=[2],
        num_blocks=[12],
        out_channels=32,
        mel_channels=80,
        mel_sample_rate=48000,
        mel_normalize_log=True,
        bottleneck=TanhBottleneck(),
    ),
    inject_depth=6,
    net_t=UNetV0,  # The model type used for diffusion upsampling
    in_channels=2,  # U-Net: number of input/output (audio) channels
    channels=[
        8,
        32,
        64,
        128,
        256,
        512,
        512,
        1024,
        1024,
    ],  # U-Net: channels at each layer
    factors=[
        1,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
        2,
    ],  # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4],  # U-Net: number of repeating items at each layer
    diffusion_t=CustomVDiffusion,  # The diffusion method used
    sampler_t=VSampler,  # The diffusion sampler used
)

model_xtransformer = ContinuousTransformerWrapper(
    dim_in=1024,
    dim_out=1024,
    max_seq_len=0,
    attn_layers=Encoder(
        dim=2048,
        depth=8,
        heads=16,
        ff_mult=2,
        attn_flash=True,
        cross_attend=False,
        zero_init_branch_output=True,
        rotary_pos_emb=True,
        ff_swish=True,
        ff_glu=True,
        use_scalenorm=True,
    ),
)

mamba_model = LatentProcessor(
    in_dim=72,
    out_dim=72,
    num_code_book=9,
    latent_dim=1024,
    num_layer=12,
)
