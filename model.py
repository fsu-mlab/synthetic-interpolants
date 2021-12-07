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
        n (int): Number of vectors of size 512 to output 
    """

    def __init__(self, cs: int, ps: int, n: int):
        """
        Params:
            cs (int): Size of the number of channels taken into the pooling layer
            ps (int): Size of the output of the pooling layer
            n (int): Number of vectors of size 512 to output 
        """
        self.n = n
        self.pool = nn.AdaptiveAvgPool2d(ps);
        self.fc = nn.Linear(7 * 7 * cs, n * 512)

    def forward(self, batch):
        batch = self.pool(batch)
        batch = torch.flatten(batch)
        batch = self.fc(batch)
        return torch.reshape(batch, self.n * 512)

class Interpolator(nn.Module):
    """
    Interpolator as inspired by the simple latent interpolation 
    model proposed by Wei et al.

    Params:
        n1 (int): Number of feature vectors to extract from the first downsample
        and efficient head
        n2 (int): Number of feature vectors to extract from the second downsample
        and efficient head
        n3 (int): Number of feature vectors to extract from the third downsample
        and efficient head
        s (int = 256): Shape of the image (s x s)
    """
    def __init__(self, n1: int, n2: int, n3: int, s: int = 256):
        """
        Params:
            n1 (int): Number of feature vectors to extract from the first downsample
            and efficient head
            n2 (int): Number of feature vectors to extract from the second downsample
            and efficient head
            n3 (int): Number of feature vectors to extract from the third downsample
            and efficient head
            s (int = 256): Shape of the image (s x s)
        """
        assert math.ceil(math.log2(s)) == math.floor(math.log2(s)), "Image size must be a power of two"

        NUM_WPLUS = int(math.log2(s)) * 2 - 2
        assert n1 + n2 + n3 == NUM_WPLUS, f"Latent vectors must add up to size {NUM_WPLUS} x 512"

        self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 48, 4, 2, 1)
        self.act = nn.ReLU()

        self.efficient1 = EfficientHead(16, 7, n1)
        self.efficient2 = EfficientHead(16, 7, n1)
        self.efficient3 = EfficientHead(16, 7, n1)

    def forward(self, batch):
        """
        In this case, w1, w2 and w3 refer to the three batches of w+
        vectors outputted by the efficient heads as opposed to the
        conventional notation for the position in the w+ matrix.
        """
        batch = self.act(self.conv1(batch))
        w1 = self.efficient1(batch) 

        batch = self.act(self.conv2(batch))
        w2 = self.efficient2(batch) 

        batch = self.act(self.conv3(batch))
        w3 = self.efficient3(batch) 

        return torch.cat([w1, w2, w3])

if __name__ == "__main__":
    # TODO: Test the network outputs 
