"""
Code for the model.
"""

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

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
    in_channels=2,  # U-Net: number of input channels
    channels=[256, 512, 1024, 1024, 1024, 1024],  # U-Net: channels at each layer
    # TODO: make this a parameter
    factors=[
        4,
        4,
        4,
        4,
        4,
        4,
    ],  # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 2, 2, 2],  # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 1, 1, 1],  # U-Net: attention enabled/disabled at each layer
    attention_heads=8,  # U-Net: number of attention heads per attention item
    attention_features=64,  # U-Net: number of attention features per attention item
    diffusion_t=CustomVDiffusion,  # The diffusion method used
    sampler_t=VSampler,  # The diffusion sampler used
)

model_medium = DiffusionModel(
    net_t=UNetV0,
    dim=1,
    in_channels=2,
    channels=[32, 32, 64, 64, 128, 128, 256, 256],
    factors=[2, 2, 2, 2, 2, 2, 2, 2],
    items=[2, 2, 2, 2, 2, 2, 4, 4],
    attentions=[0, 0, 0, 0, 0, 1, 1, 1],
    attention_heads=8,
    attention_features=64,
    diffusion_t=CustomVDiffusion,
    sampler_t=VSampler,
)
