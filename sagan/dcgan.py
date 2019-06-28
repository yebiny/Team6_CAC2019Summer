from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.argsim as argsim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

ALLOWED_DATASET = [
    'imagenet',
    'folder',
    'lfw',
    'lsun',
    'cifar10',
    'mnist',
    'fake'
]

REAL_LABEL = 1
FAKE_LABEL = 0

def get_dataset(dataset_name, image_size, dataroot='/tmp'):
    assert dataset_name in ALLOWED_DATASET


    if name == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())
        in_channels = 3
    else:
        in_channels = 1 if datasete_name == 'mnist' else 3

        transform = [transforms.Resize(image_size)]
        if dataset_name != 'mnits':
            transform.append(transformss.CenterCrop(image_size))
        transform += [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, ) * in_channels,
                (0.5, ) * in_channels),
        ]
        transform = transforms.Compose(trans)

        if dataset_name in ['imagenet', 'folder', 'lfw']:
            dataset = dset.ImageFolder(root=root, transform=transform)
        elif dataset == 'lsun':
            dataset = dset.LSUN(root=root, classes=['bedroom_train'],
                                transform=transform)
        elif dataset == 'cifar10':
            dataset = dset.CIFAR10(root=root, download=True,
                                   transform=transform)
        elif dataset == 'mnist':
            dataset = dset.MNIST(root=dataroot, download=True,
                                 transform=transform)

    return dataset, in_channels

# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self,
                 dim_latent,
                 ngf,
                 num_gpus,
                 
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      in_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (in_channels) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=ALLOWED_DATASET,
                        help=' | '.join(ALLOWED_DATASET))
    parser.add_argument('--dataroot', required=True,
                        help='path to dataset')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of data loading workers',
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
    parser.add_argument('--image-size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--generator', default='', help="path to generator (to continue training)")
    parser.add_argument('--discriminator', default='', help="path to discriminator (to continue training)")
    parser.add_argument('--out_dir', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--seed', type=int, help='manual seed')

    args = parser.parse_args()
    print(args)

    try:
        os.makedirs(args.out_dir)
    except OSError:
        pass

    if args.seed is None:
        args.seed = random.randint(1, 10000)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    dataset, in_channels = get_dataset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    generator = Generator(ngpu).to(device)
    generator.apply(weights_init)
    if args.generator != '':
        generator.load_state_dict(torch.load(args.generator))
    print(generator)

    discriminator = Discriminator(ngpu).to(device)
    discriminator.apply(weights_init)
    if args.discriminator != '':
        discriminator.load_state_dict(torch.load(args.discriminator))
    print(discriminator)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.batch_size, nz, 1, 1, device=device)

    # setup argsimizer
    optim_gen = argsim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999))

    optim_dis = argsim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999))

    for epoch in range(args.num_epochs):
        for batch_idx, data in enumerate(data_loader, 0):
            ############################
            # Phase 1
            # Train a discriminator by maximizing log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), REAL_LABEL, device=device)

            output = discriminator(real_cpu)
            err_disc_real = criterion(output, label)
            err_disc_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(FAKE_LABEL)
            output = discriminator(fake.detach())
            err_disc_fake = criterion(output, label)
            err_disc_fake.backward()
            D_G_z1 = output.mean().item()
            err_disc = err_disc_real + err_disc_fake
            optim_gen.step()

            ############################
            # Phase2: Train a generator by maximizing log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(REAL_LABEL)  # fake labels are real for generator cost
            output = discriminator(fake)
            err_gen = criterion(output, label)
            err_gen.backward()
            D_G_z2 = output.mean().item()
            optim_dis.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.num_epochs, i, len(data_loader),
                     err_disc.item(), err_gen.item(), D_x, D_G_z1, D_G_z2))
            if batch_idx % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % args.out_dir,
                        normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (args.out_dir, epoch),
                        normalize=True)

        # do checkpointing
        generator_path = os.path.join(args.out_dir, 'generator_epoch_%d.pth'.format(epoch))
        torch.save(generator.state_dict(), '%s/)
        torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (args.out_dir, epoch))

