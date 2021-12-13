# 12/10/2021
# Loss functions for the inversion network

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS

from torch.autograd import grad


class EncoderLoss(nn.Module):
    """Encoder loss for inversion"""

    def __init__(self, net: str = "alex"):
        """
        Args:
            net: String representing the network to use for LPIPS loss (default: alex)
        """
        super().__init__()
        self.lpips = LPIPS(net=net).cuda()
        self.lpips.eval()

        # Lambda values from the paper
        self.lmbda_adv = 1.
        self.lmbda_lpips = 1.

    def forward(self, inpt: torch.Tensor, target: torch.Tensor, disc: torch.Tensor):
        # Disc represents the discriminator loss
        return (
            F.l1_loss(inpt, target)
            + self.lmbda_lpips * self.lpips(inpt, target).mean()
            - self.lmbda_adv * disc.mean()
        )