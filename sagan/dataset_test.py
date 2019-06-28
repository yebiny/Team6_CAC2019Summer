import torch

from dataset import get_dataset_root
from dataset import get_dataset

def test_dataset():
    dataset = get_dataset(root=get_dataset_root())

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4)

    for batch_idx, (image, target) in enumerate(data_loader):
        print(image.shape, target.shape)
        if batch_idx == 1:
            break

    print(target)
    print(image[0])

def main():
    test_dataset()


if __name__ == '__main__':
    main()
