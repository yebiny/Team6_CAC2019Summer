import torch

from modules import Conv2dSelfAttention
from modules import CondBatchNorm2d

def test_conv2d_self_attention():
    in_channels = 64

    x = torch.zeros((5, in_channels, 6, 6)).cuda()
    print(x.shape)
    attention = Conv2dSelfAttention(in_channels=in_channels).cuda()
    print(attention)
    x = attention(x)
    print(x.shape)


def test_cond_batch_norm_2d():
    bn = CondBatchNorm2d(128, 3)
    print(bn)

def main():
    test_conv2d_self_attention()
    test_cond_batch_norm_2d()

if __name__ == '__main__':
    main()
