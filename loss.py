# 12/10/2021
# Loss functions for the inversion network

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS

from torch.autograd import grad


class InversionLoss(nn.Module):
    """Inversion loss function, including a reconstruction loss and perceptual loss"""

    def __init__(self, reconstruction: str = "l1", perceptual: str = "alex"):
        """
        Args:
            reconstruction (str): Either l1 or l2 reconstruction loss
            net (str): String representing the network to use for LPIPS loss (default: alex)
        """
        super().__init__()
        self.lpips = LPIPS(net=perceptual).cuda()
        self.lpips.eval()

        # Weights for the loss
        self.lmbda_adv = 1.
        self.lmbda_lpips = 1.

        # Determine what reconstruction loss to use 
        self.rec_loss = nn.L1Loss() if reconstruction == "l1" else nn.MSELoss()
        


    def forward(self, inpt: torch.Tensor, target: torch.Tensor, disc: torch.Tensor):
        # Disc represents the discriminator loss
        return (
            self.rec_loss(inpt, target)
            + self.lmbda_lpips * self.lpips(inpt, target).mean()
            - self.lmbda_adv * disc.mean()
        )