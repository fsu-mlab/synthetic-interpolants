# 12/10/2021
# Encoder network for FastGAN inversion,
# based heavily on https://github.com/genforce/idinvert_pytorch/blob/master/models/stylegan_encoder_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

class FastGANEncoder(nn.Module):
    """The encoder takes images with `RGB` color channels and range [-1, 1]
  as inputs, and encode the input images to z space of the GAN.
  """

    def __init__(
        self,
        resolution,
        z_space_dim=256,
        num_blocks=3,
        image_channels=3,
        encoder_channels_base=64,
        encoder_channels_max=1024,
    ):
        """Initializes the encoder with basic settings.
    Args:
      resolution: The resolution of the input image.
      z_space_dim: The dimension of the disentangled latent vector z.
        (default: 256)
      num_blocks: Number of convolutional blocks to be used. (default: 3, min: 3)
      image_channels: Number of channels of the input image. (default: 3)
      encoder_channels_base: Base factor of the number of channels used in
        residual blocks of encoder. (default: 64)
      encoder_channels_max: Maximum number of channels used in residual blocks
        of encoder. (default: 1024)
    Raises:
      ValueError: If the input `resolution` is not supported.
      ValueError: If the number of blocks is not greater than or equal to 3
    """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(
                f"Invalid resolution: {resolution}!\n"
                f"Resolutions allowed: {_RESOLUTIONS_ALLOWED}."
            )
        if num_blocks < 3 or num_blocks > int(math.log2(resolution // 8 )) + 2:
            raise ValueError(
                f"Invalid number of blocks: {num_blocks}!\n"
                f"Number of blocks must be greater than or equal to 3 and less than or equal to {int(math.log2(resolution // 8 )) + 2}."
            )

        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.num_blocks = num_blocks
        self.image_channels = image_channels
        self.encoder_channels_base = encoder_channels_base
        self.encoder_channels_max = encoder_channels_max

        in_channels = self.image_channels
        out_channels = self.encoder_channels_base

        blocks = []
        for block_idx in range(self.num_blocks):
            if block_idx == 0:
                blocks.append(
                    FirstBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                    )
                )
                blocks.append(nn.AvgPool2d(2, 2))

            elif block_idx == self.num_blocks - 1:
                # out_channels = self.z_space_dim * 2 * block_idx
                blocks.append(
                    Head(
                        in_channels=in_channels,
                        out_channels=self.z_space_dim,
                    )
                )

            else:
                blocks.append(
                    ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                    )
                )
                blocks.append(nn.AvgPool2d(2, 2))
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.encoder_channels_max)

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (
            self.image_channels,
            self.resolution,
            self.resolution,
        ):
            raise ValueError(
                f"The input image should be with shape [batch_size, "
                f"channel, height, width], where "
                f"`channel` equals to {self.image_channels}, "
                f"`height` and `width` equal to {self.resolution}!\n"
                f"But {x.shape} was received!"
            )

        x = self.network(x)
        return x


class BatchNormLayer(nn.Module):
    """Implements batch normalization layer."""

    def __init__(self, channels, gamma=False, beta=True, decay=0.9, epsilon=1e-5):
        """Initializes with basic settings.
    Args:
      channels: Number of channels of the input tensor.
      gamma: Whether the scale (weight) of the affine mapping is learnable.
      beta: Whether the center (bias) of the affine mapping is learnable.
      decay: Decay factor for moving average operations in this layer.
      epsilon: A value added to the denominator for numerical stability.
    """
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=channels,
            affine=True,
            track_running_stats=True,
            momentum=1 - decay,
            eps=epsilon,
        )
        self.bn.weight.requires_grad = gamma
        self.bn.bias.requires_grad = beta

    def forward(self, x):
        return self.bn(x)

class FirstBlock(nn.Module):
    """Implements the first block, which is a convolutional block."""

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = BatchNormLayer(channels=out_channels)
        
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Implements the residual block.
  Usually, each residual block contains two convolutional layers, each of which
  is followed by batch normalization layer and activation layer.
  """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        """Initializes the class with block settings.
    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels of the output tensor.
    Raises:
      NotImplementedError: If the input `activation_type` is not supported.
    """
        super().__init__()

        # Add shortcut if needed.
        if in_channels != out_channels:
            self.add_shortcut = True
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.bn = BatchNormLayer(channels=out_channels)
        else:
            self.add_shortcut = False
            self.identity = nn.Identity()

        hidden_channels = min(in_channels, out_channels)

        # First convolutional block.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = BatchNormLayer(channels=hidden_channels)

        # Second convolutional block.
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = BatchNormLayer(channels=out_channels)
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.add_shortcut:
            y = self.activate(self.bn(self.conv(x)))
        else:
            y = self.identity(x)
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        return x + y


class Head(nn.Module):
    """Implements the last block, which is an adaptive average pooling blockk followed
    by a dense block."""

    def __init__(
        self, in_channels, out_channels,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(
            in_features=in_channels * 49, out_features=out_channels, bias=False
        )

        self.scale = 1.0 / math.sqrt(in_channels)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
