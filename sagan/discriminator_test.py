import torch

from discriminator import DiscriminatorBlock
from discriminator import DiscriminatorOptimizedBlock

DEVICE = torch.device('cuda:0')

def test_discriminator_block():
    x = torch.randn(16, 8, 4, 4, device=DEVICE)

    args_list = [
        (8, 8, False),
        (8, 8, True),
        (8, 16, False),
        (8, 16, True),
    ]

    for args in args_list:
        block = DiscriminatorBlock(*args).to(DEVICE)
        print(block)
        print(x.shape)
        out = block(x)
        print(out.shape)

def test_discriminator_optimized_block():
    x = torch.randn(16, 8, 4, 4, device=DEVICE)
    block = DiscriminatorOptimizedBlock(8, 4).to(DEVICE)
    print(block)
    print(x.shape)
    out = block(x)
    print(out.shape)
    



def main():
    test_discriminator_block()
    test_discriminator_optimized_block()

if __name__ == '__main__':
    main()
