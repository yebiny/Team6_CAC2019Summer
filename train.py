import argparse
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from generator import Generator
from discriminator import Discriminator
from dataset import get_dataset
from dataset import get_dataset_root

def discriminator_hinge_loss(pred_real, pred_fake):
    loss_real = F.relu(1 - pred_real).mean()
    loss_fake = F.relu(1 + pred_fake).mean()
    return loss_real + loss_fake

def generator_loss(pred_fake):
    return -pred_fake.mean()

def enter_generator_phase(generator, discriminator, optim_g):
    optim_g.zero_grad()
    for each in generator.parameters():
        each.requires_grad = True
    for each in discriminator.parameters():
        each.requires_grad = False
    generator.train()
    discriminator.eval()

def enter_discriminator_phase(generator, discriminator, optim_d):
    optim_d.zero_grad()
    for each in generator.parameters():
        each.requires_grad = False
    for each in discriminator.parameters():
        each.requires_grad = True
    generator.eval()
    discriminator.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--num-epochs', default=5, type=int)
    parser.add_argument('--dim-latent', default=128, type=int)
    parser.add_argument('--generator-lr', default=0.0001, type=float)
    parser.add_argument('--discriminator-lr', default=0.0004, type=float)
    parser.add_argument('--discriminator-update-freq', default=3, type=int)
    parser.add_argument('--num-channels', default=3, type=int)
    parser.add_argument('--num-classes', default=5, type=int)
    parser.add_argument('--seed', default=12509, type=int)
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    # NOTE for reporducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = get_dataset(get_dataset_root())
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             drop_last=True)

    # FIXME
    device = torch.device('cuda:0')
    
    generator = Generator(dim_latent=args.dim_latent,
                          out_channels=args.num_channels,
                          num_classes=args.num_classes).to(device)
    print(generator)

    discriminator = Discriminator(in_channels=args.num_channels,
                                  num_classes=args.num_classes).to(device)
    print(discriminator)

    optim_g = optim.Adam(generator.parameters(),
                         lr=args.generator_lr,
                         betas=(0, 0.9))

    optim_d = optim.Adam(discriminator.parameters(),
                         lr=args.discriminator_lr,
                         betas=(0, 0.9))

    enter_discriminator_phase(generator, discriminator, optim_d)
    for epoch in range(args.num_epochs):
        print('[Epoch {:>3d} / {:>3d}]'.format(epoch, args.num_epochs))
        for batch_idx, (x, y) in enumerate(data_loader, 1):
            # NOTE Discriminator training phase
            x = x.to(device)
            y = y.to(device)

            if batch_idx % args.discriminator_update_freq == 1:
                enter_discriminator_phase(generator, discriminator, optim_d)

            # input noise = latent vcector
            z = torch.randn(args.batch_size, args.dim_latent, device=device)
            # fake image = x sampled from generator
            x_fake = generator(z, y)

            y_hat_real = discriminator(x, y)
            y_hat_fake = discriminator(x_fake.detach(), y)

            loss_d = discriminator_hinge_loss(y_hat_real, y_hat_fake)
            loss_d.backward()
            optim_d.step()

            # NOTE Generator training phase
            if batch_idx % args.discriminator_update_freq == 0:
                enter_generator_phase(generator, discriminator, optim_g)
 
                # input noize = latent vector
                z = torch.randn(args.batch_size, args.dim_latent, device=device)
                y = torch.multinomial(torch.ones(args.num_classes),
                                      args.batch_size,
                                      replacement=True).to(device)

                # x sampled from model
                x_fake = generator(z, y)
                y_hat_fake = discriminator(x_fake, y)

                loss_g = generator_loss(y_hat_fake)
                loss_g.backward()
                optim_g.step()

                print('  Batch {:>3d}/{:d} Loss Discriminator: {:.3f} Loss Generator: {:.3f}'.format(
                    batch_idx , len(data_loader), loss_d.item(), loss_g.item()))


if __name__ == '__main__':
    main()
