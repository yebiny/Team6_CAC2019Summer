import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Interpolation(nn.Module):
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Interpolation, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor,
                             self.mode, self.align_corners)


class Conv2dSelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, k=8):
        super(Conv2dSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.k = k

        self.bottleneck_channels = int(in_channels / k)
        self.g_channels = int(in_channels / 2)

        self.conv_theta = spectral_norm(nn.Conv2d(
            in_channels=in_channels, out_channels=self.bottleneck_channels,
            kernel_size=1, stride=1, padding=0))

        self.conv_phi = spectral_norm(nn.Conv2d(
            in_channels=in_channels, out_channels=self.bottleneck_channels,
            kernel_size=1, stride=1, padding=0))

        self.conv_g = spectral_norm(nn.Conv2d(
            in_channels=in_channels, out_channels=self.g_channels,
            kernel_size=1, stride=1, padding=0))

        self.conv_out = spectral_norm(nn.Conv2d(
            in_channels=self.g_channels,
            out_channels=in_channels,
            kernel_size=1, stride=1, padding=0))

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        num_locations = height * width

        # f - theta
        theta = self.conv_theta(x)
        theta = theta.view(-1, self.bottleneck_channels, num_locations)
        theta = theta.permute(0, 2, 1).contiguous()

        # g - phi
        phi = self.conv_phi(x)
        # phi = F.interpolate(phi, scale_factor=0.5)
        phi = F.avg_pool2d(phi, kernel_size=2, stride=2)
        phi = phi.view(-1, self.bottleneck_channels, int(num_locations / 4))
        phi = phi.contiguous()

        # h - g
        g = self.conv_g(x)
        g = F.avg_pool2d(g, kernel_size=2, stride=2)
        g = g.view(-1, self.g_channels, int(num_locations / 4))
        g = g.contiguous()

        # Attention
        attention_logits = torch.bmm(theta, phi)
        attention = attention_logits.softmax(dim=2)
        attention = attention.permute(0, 2, 1).contiguous()
        # print(attention.shape)
        # print(g.shape)

        # Attention
        out = torch.bmm(g, attention)
        out = out.view(-1, self.g_channels, height, width).contiguous()
        out = self.conv_out(out)

        # Out
        out = self.gamma * out + x
        return out


class CondBatchNorm2d(nn.Module):
    '''
    Conditional batch normalization
    https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    '''
    def __init__(self, num_features, num_classes):
        super(CondBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)

        self.embedding = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embedding.weight.data[:, : num_features].fill_(1.)
        self.embedding.weight.data[:, num_features: ].zero_()

    def forward(self, x, y):
        out = self.batch_norm(x)
        gamma, beta = self.embedding(y).chunk(2, 1)

        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta =  beta.view(-1, self.num_features, 1, 1)

        out = gamma * out + beta
        return out

