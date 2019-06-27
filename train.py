import torch
import torch.nn.functional as F

from generator import Generator
from discriminator import Discriminator
from dataset import get_data_loader

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
    parser.add_argument('--generator-lr', default=0.0001, type=float)
    parser.add_argument('--generator-lr', default=0.0004, type=float)
    parser.add_argument('--discriminator-update-freq', default=5, type=int)
    args = parser.parse_args()

    dataset = get_dataset()
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=True)
        
    generator = Generator()
    discriminator = Discriminator()

    optim_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    optim_d = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0, 0.9))

    for epoch in range(args.num_epochs):
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
                y = torch.multinomial(torch.ones(num_classes), args.batch_size,
                                      replacement=True)

                # x sampled from model
                x_fake = generator(z, y)
                y_hat_fake = discriminator(x_fake, y)

                loss_g = generator_loss(y_hat_fake)
                loss_g.backward()
                optim_g.step()
