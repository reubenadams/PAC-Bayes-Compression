import os

import torch
from torchvision import datasets, transforms


def get_datasets(dataset_name, new_size=None):

    if dataset_name == "MNIST":
        data_root = "./data/MNIST"
        if new_size is None:
            new_size = (28, 28)
            if new_size[0] > 28 or new_size[1] > 28:
                raise ValueError(
                    f"New MNIST size {new_size} should not be larger than original size 28x28."
                )

    elif dataset_name == "CIFAR10":
        data_root = "./data/CIFAR10"
        if new_size is None:
            new_size = (32, 32)
            if new_size[0] > 32 or new_size[1] > 32:
                raise ValueError(
                    f"New CIFAR10 size {new_size} should not be larger than original size 32x32."
                )

    data_dir = os.path.join(data_root, f"{new_size[0]}x{new_size[1]}")

    try:

        train = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
        test = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)

    except FileNotFoundError:

        transform = transforms.Compose(
            [
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        if dataset_name == "MNIST":

            train = datasets.MNIST(
                root=data_root,
                train=True,
                download=True,
                transform=transform,
            )
            test = datasets.MNIST(
                root=data_root,
                train=False,
                download=True,
                transform=transform,
            )

        elif dataset_name == "CIFAR10":

            train = datasets.CIFAR10(
                root=data_root,
                train=True,
                download=True,
                transform=transform,
            )
            test = datasets.CIFAR10(
                root=data_root,
                train=False,
                download=True,
                transform=transform,
            )

        os.makedirs(data_dir, exist_ok=True)
        torch.save(train, os.path.join(data_dir, "train.pt"))
        torch.save(test, os.path.join(data_dir, "test.pt"))

    return train, test


def get_dataloaders(
    dataset_name, batch_size, train_size=None, test_size=None, new_size=None
):

    train, test = get_datasets(dataset_name, new_size)

    if train_size is not None:
        train = torch.utils.data.Subset(train, range(train_size))
    if test_size is not None:
        test = torch.utils.data.Subset(test, range(test_size))

    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False)

    return train_loader, test_loader


def get_B(data_loader):
    max_fro_norm = torch.tensor(0.0)
    for data, _ in data_loader:
        fro_norms = torch.linalg.matrix_norm(data, ord="fro")
        max_fro_norm = max(max_fro_norm, fro_norms.max())
    return max_fro_norm
