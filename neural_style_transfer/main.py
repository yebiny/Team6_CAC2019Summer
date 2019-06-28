from __future__ import print_function

import os
import argparse
import copy

import matplotlib as mpl
mpl.use("Agg")
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

from losses import StyleLoss
from losses import ContentLoss
from modules import Normalization 

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def load_image(image_name, transform, device):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, ax, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = unloader(image)
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, device,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]
    print(model)

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, device, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        style_img,
        content_img,
        device)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--style', required=True)
    parser.add_argument('-c', '--content', required=True)
    # parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('--image-size', default=512, type=int)
    args = parser.parse_args()

    if args.output is None:
        style_basename = os.path.splitext(os.path.basename(args.style))[0]
        content_basename = os.path.splitext(os.path.basename(args.content))[0]
        args.output = 'style-{}_content-{}.png'.format(
            style_basename, content_basename)

    device = torch.device("cuda:0")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    style_img = load_image(args.style, transform, device)
    content_img = load_image(args.content, transform, device)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    # desired depth layers to compute style/content losses :
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, device)

    fig, axarr = plt.subplots(ncols=3, figsize=(24, 8))
    imshow(style_img, axarr[0], title='Style Image')
    imshow(content_img, axarr[1], title='Content Image (Input Image)')
    imshow(output, axarr[2], title='Output Image')
    fig.savefig(args.output)


if __name__ == '__main__':
    main()
