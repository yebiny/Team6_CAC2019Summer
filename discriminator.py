import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from modules import Interpolation

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscriminatorBlock, self).__init__()

        # NOTE residual function
        residual_function = [
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1))
        ]
        if downsample:
            residual_function.append(Interpolation(scale_factor=0.5))
        self.residual_function = nn.Sequential(*residual_function)

        # NOTE identity mapping
        identity_mapping = []
        if in_channels != out_channels:
            identity_mapping.append(spectral_norm(nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)))
        if downsample:
            identity_mapping.append(Interpolation(scale_factor=0.5))

        if len(identity_mapping) == 2:
            self.identity_mapping = nn.Sequential(*identity_mapping)
        elif len(identity_mapping) == 1:
            self.identity_mapping = identity_mapping[0]
        else:
            self.identity_mapping = nn.Identity()
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

    def forward(self, x):
        # assert self.in_channels == x.size(1)
        identity = self.identity_mapping(x)
        residual = self.residual_function(x)
        return identity + residual



        
class DiscriminatorOptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorOptimizedBlock, self).__init__()

        self.residual_function = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, stride=1, padding=1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(out_channels, out_channels,
                                    kernel_size=3, stride=1, padding=1)),
            Interpolation(scale_factor=0.5))

        self.identity_mapping = nn.Sequential(
            Interpolation(scale_factor=0.5),
            spectral_norm(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, stride=1, padding=0)))

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # assert self.in_channels == x.size(1)
        identity = self.identity_mapping(x)
        residual = self.residual_function(x)
        return identity + residual


class Discriminator(nn.Module):
    '''
    Projection-based discriminator
    '''
    def __init__(self, in_channels, num_classes):
        super(Discriminator, self).__init__()
            # FIXME fix name
        ddim = 1

        self.stem = nn.Sequential(
            DiscriminatorOptimizedBlock(in_channels, ddim),
            DiscriminatorBlock(ddim, 2 * ddim),
            Conv2dSelfAttention(2 * ddim),
            DiscriminatorBlock(2 * ddim, 4 * ddim),
            DiscriminatorBlock(4 * ddim, 8 * ddim),
            DiscriminatorBlock(8 * ddim, 16 * ddim),
            DiscriminatorBlock(16 * ddim, 16 * ddim, downsample=False),
            nn.ReLU(inplace=True))

        self.linear = spectral_norm(nn.Linear(
            in_features=16 * ddim, out_features=1))

        self.embedding = spectral_norm(nn.Embedding(
            num_classes, 16 * ddim))

        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, image, target):
        out = self.stem(image)
        out = out.sum(dim=(2, 3)).squeeze()
        logits = self.linear(out)

        # TODO implement projection
        # https://arxiv.org/pdf/1802.05637.pdf
        h_labels = self.embedding(target)
        projection = h_labels.mul(out).sum(dim=1)
        return logits + projection
