import os

import torchvision
from torchvision import transforms

def get_dataset_root():
    hostname = os.environ['HOSTNAME']
    if hostname == 'gate2.sscc.uos.ac.kr':
        root = '/scratch/seyang/kias-cac/cat/'
    elif hostname == 'dgx':
        root = '/scratch/seyang/cat'
    else:
        raise NotImplementedError
    return root


def get_dataset(root,
                size=(128, 128),
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return torchvision.datasets.ImageFolder(
        root=root, transform=transform)
