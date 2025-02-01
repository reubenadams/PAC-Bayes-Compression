import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset


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
        train = Subset(train, range(train_size))
    if test_size is not None:
        test = Subset(test, range(test_size))

    train_loader = DataLoader(train, batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size, shuffle=False)

    return train_loader, test_loader


def get_B(data_loader):
    max_fro_norm = torch.tensor(0.0)
    for data, _ in data_loader:
        fro_norms = torch.linalg.matrix_norm(data, ord="fro")
        max_fro_norm = max(max_fro_norm, fro_norms.max())
    return max_fro_norm


class RandomDomainDataset(Dataset):

    def __init__(self, data_size, sample_size):
        self.num_pixels = data_size[0] * data_size[1]
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    # We return idx just so the DataLoader doesn't complain. Note normalization is just to get the right range, it's nothing to do with wanting to match the distribution.
    def __getitem__(self, idx):
        return (torch.rand((self.num_pixels)) - 0.5) / 0.5, idx


def get_rand_domain_dataloader(data_size, sample_size, batch_size):
    return DataLoader(
        RandomDomainDataset(data_size, sample_size), batch_size, shuffle=True
    )


def get_epsilon_mesh(epsilon, data_size, device):
    cell_width = torch.sqrt(torch.tensor(2.0, device=device)) * epsilon
    num_cells = int(torch.ceil(1 / cell_width))
    actual_cell_width = 1 / num_cells
    actual_epsilon = actual_cell_width / torch.sqrt(torch.tensor(2.0, device=device))
    num_pixels = data_size[0] * data_size[1]
    print(
        f"Creating mesh with {num_cells}^{num_pixels} = {num_cells ** num_pixels} elements."
    )
    mesh = torch.cartesian_prod(
        *[torch.linspace(0, 1, num_cells + 1, device=device)] * num_pixels
    )
    return mesh, actual_epsilon, actual_cell_width


# TODO: I think this is not going to work because it's only returning x's and not (fake) y's.
def get_epsilon_mesh_dataloader(epsilon, data_size, batch_size):
    mesh, eps, width = get_epsilon_mesh(epsilon, data_size)
    mesh_dataloader = DataLoader(mesh, batch_size, shuffle=True)
    return mesh_dataloader, eps, width
