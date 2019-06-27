import torch

from modules import Conv2dSelfAttention
from modules import CondBatchNorm2d

def test_conv2d_self_attention():
    x = torch.zeros((64, 32, 8, 8)).cuda()
    print(x.shape)
    attention = Conv2dSelfAttention(in_channels=32).cuda()
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
