"""
Discriminators
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from audioenhancer.model.audio_ae.encoder import Encoder1d
from audioenhancer.model.audio_ae.utils import default


class Discriminator1d(nn.Module):
    """
        A class used to represent the Discriminator in a Generative Adversarial Network (GAN).

        ...

        Attributes
        ----------
        discriminator : Encoder1d
            an instance of the Encoder1d class which is used as the discriminator in the GAN
        use_loss : list
            a list of boolean values indicating whether to extract the discrimination loss from each layer

        Methods
        -------
        forward(true: Tensor, fake: Tensor, with_info: bool = False) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Dict]]:
            Computes the forward pass of the discriminator.
        """
    def __init__(self, use_loss: Optional[Sequence[bool]] = None, **kwargs):
        """
        Constructs all the necessary attributes for the Discriminator1d object.
        Parameters
        ----------
            use_loss : list, optional
                a list of boolean values indicating whether to extract the discrimination loss from each layer (default is None, which means all layers are used)
            **kwargs : dict
                arbitrary keyword arguments
        """
        super().__init__()
        self.discriminator = Encoder1d(**kwargs)
        num_layers = self.discriminator.num_layers
        # By default we activate discrimination loss extraction on all layers
        self.use_loss = default(use_loss, [True] * num_layers)
        # Check correct length
        msg = f"use_loss length must match the number of layers ({num_layers})"
        assert len(self.use_loss) == num_layers, msg

    def forward(
        self, true: Tensor, fake: Tensor, with_info: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Dict]]:
        """
        Computes the forward pass of the discriminator.

        Parameters
        ----------
            true : Tensor
                the real data
            fake : Tensor
                the generated data
            with_info : bool, optional
                whether to return additional info (default is False)

        Returns
        -------
            Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Dict]]
                the generator loss, the discriminator loss, and optionally a dictionary with additional info
        """
        # Get discriminator outputs for true/fake scores
        _, info_true = self.discriminator(true, with_info=True)
        _, info_fake = self.discriminator(fake, with_info=True)

        # Get all intermediate layer features (ignore input)
        xs_true = info_true["xs"][1:]
        xs_fake = info_fake["xs"][1:]

        loss_gs, loss_ds, scores_true, scores_fake = [], [], [], []

        for use_loss, x_true, x_fake in zip(self.use_loss, xs_true, xs_fake):
            if use_loss:
                # Half the channels are used for scores, the other for features
                score_true, feat_true = x_true.chunk(chunks=2, dim=1)
                score_fake, feat_fake = x_fake.chunk(chunks=2, dim=1)
                # Generator must match features with true sample and fool discriminator
                loss_gs += [F.l1_loss(feat_true, feat_fake) - score_fake.mean()]
                # Discriminator must give high score to true samples, low to fake
                loss_ds += [((1 - score_true).relu() + (1 + score_fake).relu()).mean()]
                # Save scores
                scores_true += [score_true.mean()]
                scores_fake += [score_fake.mean()]

        # Average all generator/discriminator losses over all layers
        loss_g = torch.stack(loss_gs).mean()
        loss_d = torch.stack(loss_ds).mean()

        info = dict(scores_true=scores_true, scores_fake=scores_fake)

        return (loss_g, loss_d, info) if with_info else (loss_g, loss_d)