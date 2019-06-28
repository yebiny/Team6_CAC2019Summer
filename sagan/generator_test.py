import torch

from generator import GeneratorBlock
from generator import Generator

BATCH_SIZE = 16
DIM_LATENT = 32
NUM_CLASSES = 5
DEVICE = torch.device('cuda:0')

LATENT_VECTOR = torch.randn(BATCH_SIZE, DIM_LATENT, device=DEVICE)
TARGET = torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE, ), device=DEVICE)

def test_local_block():
    batch_size = 16
    in_channels = 32
    num_classes = 5
    device = torch.device('cuda:0')

    out_channels = int(in_channels / 2)

    x = torch.randn(batch_size, in_channels, 8, 8, device=device)
    y = torch.randint(low=0, high=num_classes, size=(batch_size, ), device=device)

    block = GeneratorBlock(in_channels=in_channels, out_channels=out_channels,
                       num_classes=num_classes).to(device)

    print(block)

    print(x.shape)
    h = block(x, y)
    print(h.shape)

def test_generator():
    gen = Generator(dim_latent=DIM_LATENT, num_classes=NUM_CLASSES).to(DEVICE)
    print(gen)
    fake = gen(LATENT_VECTOR, TARGET)
    print('Fake: {}'.format(fake.shape))

def main():
    test_local_block()
    test_generator()


if __name__ == '__main__':
    main()
