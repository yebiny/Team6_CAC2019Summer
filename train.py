import os
import sys
import argparse
import random
import numpy as np
from datetime import datetime

if sys.version_info.major == 2:
    from pathlib2 import Path
else:
    from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.utils import save_image

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

def enter_generator_phase(generator, discriminator, generator_optimizer):
    generator_optimizer.zero_grad()
    for each in generator.parameters():
        each.requires_grad = True
    for each in discriminator.parameters():
        each.requires_grad = False
    generator.train()
    discriminator.eval()

def enter_discriminator_phase(generator, discriminator, discriminator_optimizer):
    discriminator_optimizer.zero_grad()
    for each in generator.parameters():
        each.requires_grad = False
    for each in discriminator.parameters():
        each.requires_grad = True
    generator.eval()
    discriminator.train()

def save_state_dict(generator,
                    discriminator,
                    generator_optimizer,
                    discriminator_optimizer,
                    epoch,
                    root):
    basename = 'checkpoint_epoch-{:0>4d}'.format(epoch)
    path = str(root.joinpath(basename))

    state_dict = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'generator_optimizer': generator_optimizer.state_dict(),
        'discriminator_optimizer': discriminator_optimizer.state_dict(),
    }

    torch.save(state_dict, path)


def sample_image(generator,
                 dim_latent,
                 num_classes,
                 epoch,
                 root,
                 num_samples,
                 device):

    directory = root.joinpath('epoch-{:0>4d}')
    directory.mkdir()

    for yi in range(num_classes):
        npz_path = str(directory.joinpath('sample-{}.npz'.format(yi)))
        image_path = str(directory.joinpath('image-{}.png'.format(yi)))

        z = torch.randn(num_samples, dim_latent, device=device)
        y = yi * torch.ones(num_samples, device=device)
        x_fake = generator(z, y)

        x_fake = 0.5 * x_fake + 0.5

        save_image(x_fake, image_path, nrows=4)

        x_fake = x_fake.detach().cpu().numpy()
        np.savez(npz_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--num-epochs', default=5, type=int)
    parser.add_argument('--dim-latent', default=128, type=int)
    parser.add_argument('--generator-lr', default=0.0001, type=float)
    parser.add_argument('--discriminator-lr', default=0.0002, type=float)
    parser.add_argument('--discriminator-update-freq', default=3, type=int)
    parser.add_argument('--num-channels', default=3, type=int)
    parser.add_argument('--num-classes', default=5, type=int)
    parser.add_argument('--seed', default=12509, type=int)
    parser.add_argument('--num-samples', default=16, type=int)

    default_name = 'sagan_{}_{}'.format(
        os.environ['USER'],
        datetime.now().strftime('%y%m%d-%H%M%S'))
    default_out_dir = os.path.join('/tmp/', default_name)
    parser.add_argument('--out-dir', default=default_out_dir)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir()

    ckpt_dir = out_dir.joinpath('checkpoint')
    ckpt_dir.mkdir()

    sample_dir = out_dir.joinpath('sample')
    sample_dir.mkdir()

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

    generator_optimizer = optim.Adam(generator.parameters(),
                                     lr=args.generator_lr,
                                     betas=(0, 0.9))

    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=args.discriminator_lr,
                                         betas=(0, 0.9))

    enter_discriminator_phase(generator, discriminator, discriminator_optimizer)
    for epoch in range(args.num_epochs):
        print('[Epoch {:>3d} / {:>3d}]'.format(epoch, args.num_epochs))
        for batch_idx, (x, y) in enumerate(data_loader, 1):
            # NOTE Discriminator training phase
            x = x.to(device)
            y = y.to(device)

            if batch_idx % args.discriminator_update_freq == 1:
                enter_discriminator_phase(generator, discriminator,
                                          discriminator_optimizer)

            # input noise = latent vcector
            z = torch.randn(args.batch_size, args.dim_latent, device=device)
            # fake image = x sampled from generator
            x_fake = generator(z, y)

            y_hat_real = discriminator(x, y)
            y_hat_fake = discriminator(x_fake.detach(), y)

            loss_d = discriminator_hinge_loss(y_hat_real, y_hat_fake)
            loss_d.backward()
            discriminator_optimizer.step()

            # NOTE Generator training phase
            if batch_idx % args.discriminator_update_freq == 0:
                enter_generator_phase(generator, discriminator,
                                      generator_optimizer)
 
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
                generator_optimizer.step()

                print('  Batch {:>3d}/{:d} Loss Discriminator: {:.3f} Loss Generator: {:.3f}'.format(
                    batch_idx , len(data_loader), loss_d.item(), loss_g.item()))

        save_state_dict(generator, discriminator,
                        generator_optimizer, discriminator_optimizer,
                        epoch, ckpt_dir)

        sample_image(generator,
                     args.dim_latent,
                     args.num_classes,
                     epoch,
                     sample_dir,
                     args.num_samples,
                     device)



if __name__ == '__main__':
    main()
