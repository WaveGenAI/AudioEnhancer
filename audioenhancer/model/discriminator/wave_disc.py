import torch
from torch import nn

import math
from audioenhancer.constants import MAX_AUDIO_LENGTH
from audioenhancer.model.encoder import Encoder
from audioenhancer.model.latent import Latent


class Pooler(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Discriminator(nn.Module):
    def __init__(self, latent_dim, num_channels, strides=(2, 4, 5, 8)):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(C=num_channels, D=latent_dim, strides=strides)
        self.latent = Latent(d_model=latent_dim, intermediate_dim=1024, num_layers=4)
        self.pooler = Pooler(d_model=latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)
        self.init_weights()
        torch.cuda.empty_cache()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e = self.encoder(x)
        l = self.latent(e).permute(0, 2, 1)
        p = self.pooler(l)
        logits = self.classifier(p)
        return logits
