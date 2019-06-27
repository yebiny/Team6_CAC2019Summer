import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from modules import Conv2dSelfAttention
from modules import CondBatchNorm2d



class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(LocalBlock, self).__init__()

        self.bn0 = CondBatchNorm2d(in_channels, num_classes)
        self.bn1 = CondBatchNorm2d(out_channels, num_classes)

        self.conv0 = spectral_norm(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1))

        self.conv1 = spectral_norm(nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1))

        self.conv2 = spectral_norm(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0))

    def forward(self, x, y):
        h = self.bn0(x, y)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv0(h)
        h = self.bn1(h, y)
        h = F.relu(h)
        h = self.conv1(h)

        # identity mapping
        identity = F.interpolate(x, scale_factor=2, mode='nearest')
        identity = self.conv2(identity)
        
        return h + identity


class Generator(nn.Module):
    def __init__(self,
                 dim_latent,
                 num_classes,
                 out_channels=3):
        super(Generator, self).__init__()
        unit = 16
        self.chw = (unit * 16, 4, 4)

        self.linear = spectral_norm(nn.Linear(
            in_features=dim_latent, out_features=np.prod(self.chw)))

        self.local_block_1 = LocalBlock(unit * 16, unit * 16, num_classes)
        self.local_block_2 = LocalBlock(unit * 16, unit * 8, num_classes)

        self.local_block_3 = LocalBlock(unit * 8, unit * 4, num_classes)
        self.global_block = Conv2dSelfAttention(unit * 4)

        self.local_block_4 = LocalBlock(unit * 4, unit * 2, num_classes)
        self.local_block_5 = LocalBlock(unit * 2, unit, num_classes)


        self.top = nn.Sequential(
            nn.BatchNorm2d(unit),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(
                unit, out_channels,
                kernel_size=3, stride=1, padding=1)),
            nn.Tanh())


    def forward(self, x, y):
        x = self.linear(x)
        x = x.view(-1, *self.chw)

        x = self.local_block_1(x, y)
        x = self.local_block_2(x, y)
        x = self.local_block_3(x, y)
        x = self.global_block(x)
        x = self.local_block_4(x, y)
        x = self.local_block_5(x, y)

        return self.top(x)
        
