import random

import torch
from torch import nn, Tensor

from audioenhancer.model.audio_ae.mamba import MambaBlock


class Expert(nn.Module):
    def __init__(self, num_mamba_layer: nn.Module, num_expert, config):
        super().__init__()
        self.experts = nn.ModuleList()
        for _ in range(num_expert - 1):
            layers = nn.Sequential()
            for i in range(num_mamba_layer):
                layers.append(MambaBlock(config))
            self.experts.append(layers)


    def forward(self, x, expert_id: Tensor = None):
        if expert_id is None:
            return torch.stack([x] + [expert(x) for expert in self.experts])
        else:
            out = []
            for i, expert_idx in enumerate(expert_id):
                if random.random() < 0.1:
                    expert_idx = random.randint(1, 4)
                if expert_idx == 0:
                    out.append(x[i])
                else:
                    out.append(self.experts[expert_idx - 1](x[i].unsqueeze(0)).squeeze(0))
            return torch.stack(out)
