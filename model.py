import torch
import torch.nn as nn

import math

class EfficientHead(nn.Module):
    """
    Efficient Head as inspired by the simple latent interpolation
    model proposed by Wei et al.

    Params:
        cs (int): Size of the number of channels taken into the pooling layer
        ps (int): Size of the output of the pooling layer
        n (int): Number of output latent vectors of size w_dim
        w_dim (int): Size of the w+ vectors 
    """

    def __init__(self, cs: int, ps: int, n: int, w_dim):
        """
        Params:
            cs (int): Size of the number of channels taken into the pooling layer
            ps (int): Size of the output of the pooling layer
            n (int): Number of output latent vectors of size w_dim
            w_dim (int): Size of the w+ vectors 
        """
        super().__init__()
        self.n = n
        self.w_dim = w_dim
        self.pool = nn.AdaptiveAvgPool2d(ps)
        self.fc = nn.Linear(ps * ps * cs, n * w_dim)

    def forward(self, batch):
        bs = batch.shape[0]
        batch = self.pool(batch)
        batch = torch.flatten(batch, start_dim=1)
        batch = self.fc(batch)
        return torch.reshape(batch, (bs, self.n, self.w_dim))

class Inverter(nn.Module):
    """
    Inverter as inspired by the simple latent interpolation 
    model proposed by Wei et al.

    Params:
        n1 (int): Number of feature vectors to extract from the first downsample
        and efficient head
        n2 (int): Number of feature vectors to extract from the second downsample
        and efficient head
        n3 (int): Number of feature vectors to extract from the third downsample
        and efficient head
        w_dim (int): Size of the w+ vectors 
        s (int = 256): Shape of the image (s x s)
    """
    def __init__(self, n1: int, n2: int, n3: int, w_dim: int, s: int = 256):
        """
        Params:
            n1 (int): Number of feature vectors to extract from the first downsample
            and efficient head
            n2 (int): Number of feature vectors to extract from the second downsample
            and efficient head
            n3 (int): Number of feature vectors to extract from the third downsample
            and efficient head
            w_dim (int): Size of the w+ vectors 
            s (int = 256): Shape of the image (s x s)
        """
        super().__init__()
        assert math.ceil(math.log2(s)) == math.floor(math.log2(s)), "Image size must be a power of two"

        NUM_WPLUS = int(math.log2(s)) * 2 - 2
        assert n1 + n2 + n3 == NUM_WPLUS, f"Latent vectors must add up to size {NUM_WPLUS} x 512"

        self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 48, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)

        self.act = nn.ReLU()

        self.efficient1 = EfficientHead(16, 7, n1, w_dim)
        self.efficient2 = EfficientHead(32, 5, n2, w_dim)
        self.efficient3 = EfficientHead(48, 3, n3, w_dim)

    def forward(self, batch):
        """
        In this case, w1, w2 and w3 refer to the three batches of w+
        vectors outputted by the efficient heads as opposed to the
        conventional notation for the position in the w+ matrix.
        """
        batch = self.act(self.bn1(self.conv1(batch)))
        w1 = self.efficient1(batch) 

        batch = self.act(self.bn2(self.conv2(batch)))
        w2 = self.efficient2(batch) 

        batch = self.act(self.bn3(self.conv3(batch)))
        w3 = self.efficient3(batch) 

        return torch.cat([w1, w2, w3], dim=1)

if __name__ == "__main__":
    pass
    # TODO: Test the network outputs 