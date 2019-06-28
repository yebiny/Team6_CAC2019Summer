import torch
import torch.nn as nn

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class Conv2dSelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, k=8):
        super(Conv2dSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.k = k

        self.bottleneck_channels = int(in_channels / k)
        self.g_channels = int(in_channels / 2)

        self.conv_theta = nn.Conv2d(
            in_channels=in_channels, out_channels=self.bottleneck_channels,
            kernel_size=1, stride=1, padding=0)

        self.conv_phi = nn.Conv2d(
            in_channels=in_channels, out_channels=self.bottleneck_channels,
            kernel_size=1, stride=1, padding=0)

        self.conv_g = nn.Conv2d(
            in_channels=in_channels, out_channels=self.g_channels,
            kernel_size=1, stride=1, padding=0)

        self.conv_out = nn.Conv2d(
            in_channels=self.g_channels,
            out_channels=in_channels,
            kernel_size=1, stride=1, padding=0)

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
        return self.conv_out(out)
