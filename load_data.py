import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from mnist1d.data import get_dataset_args, get_dataset


def get_datasets(dataset_name, new_size=None):

    valid_datasets = ["MNIST", "CIFAR10", "MNIST1D"]
    data_root = f"./data/{dataset_name}"

    if dataset_name == "MNIST":
        if new_size is None:
            new_size = (28, 28)
            if new_size[0] > 28 or new_size[1] > 28:
                raise ValueError(
                    f"New MNIST size {new_size} should not be larger than original size 28x28."
                )
        data_dir = os.path.join(data_root, f"{new_size[0]}x{new_size[1]}")

    elif dataset_name == "CIFAR10":
        if new_size is None:
            new_size = (32, 32)
            if new_size[0] > 32 or new_size[1] > 32:
                raise ValueError(
                    f"New CIFAR10 size {new_size} should not be larger than original size 32x32."
                )
        data_dir = os.path.join(data_root, f"{new_size[0]}x{new_size[1]}")

    elif dataset_name == "MNIST1D":
        if new_size is None:
            new_size = (40,)
        else:
            if new_size != (40,):
                raise ValueError(f"MNIST1D does not support resizing.")
        data_dir = os.path.join(data_root, f"{new_size[0]}")

    else:
        raise ValueError(
            f"Invalid dataset name: {dataset_name} should be one of {valid_datasets}."
        )

    try:
        print(f"Loading data from {data_dir}.")
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

        elif dataset_name == "MNIST1D":
            os.makedirs(data_root, exist_ok=True)
            path = os.path.join(data_root, "mnist1d_data.pkl")
            args = get_dataset_args()
            data = get_dataset(args, path=path, download=True)
            x_train = torch.tensor(data["x"], dtype=torch.float32)
            x_test = torch.tensor(data["x_test"], dtype=torch.float32)
            y_train, y_test = torch.tensor(data["y"]), torch.tensor(data["y_test"])
            train = CustomDataset(x_train, y_train)
            test = CustomDataset(x_test, y_test)

        os.makedirs(data_dir, exist_ok=True)
        torch.save(train, os.path.join(data_dir, "train.pt"))
        torch.save(test, os.path.join(data_dir, "test.pt"))

    return train, test


def get_dataloaders(
    dataset_name, batch_size, train_size=None, test_size=None, new_size=None, device="cpu"
):

    train, test = get_datasets(dataset_name, new_size)

    if train_size is not None:
        train = Subset(train, range(train_size))
    if test_size is not None:
        test = Subset(test, range(test_size))

    train.data = train.data.to(device)
    train.targets = train.targets.to(device)
    test.data = test.data.to(device)
    test.targets = test.targets.to(device)

    train_loader = DataLoader(train, batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size, shuffle=False)

    return train_loader, test_loader


def get_B(data_loader):
    max_fro_norm = torch.tensor(0.0)
    for data, _ in data_loader:
        fro_norms = torch.linalg.matrix_norm(data, ord="fro")
        max_fro_norm = max(max_fro_norm, fro_norms.max())
    return max_fro_norm


class CustomDataset(Dataset):
        
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class RandomDomainDataset(Dataset):

    def __init__(self, data_shape, sample_size):
        self.num_pixels = data_shape[0] * data_shape[1]
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    # We return idx just so the DataLoader doesn't complain.
    def __getitem__(self, idx):
        return (
            torch.rand((self.num_pixels)) - 0.5
        ) / 0.5, idx  # TODO: To do this properly we need to pass in the actual range of the data, i.e. we need to know the mean and std of the data.


class MeshDomainDataset(Dataset):

    def __init__(self, data_shape, epsilon):
        self.num_pixels = data_shape[0] * data_shape[1]
        self.mesh, self.actual_epsilon, self.actual_cell_width = get_epsilon_mesh(
            epsilon, data_shape, device="cpu"
        )
        self.mesh = (
            self.mesh - 0.5
        ) / 0.5  # TODO: To do this properly we need to pass in the actual range of the data, i.e. we need to know the mean and std of the data.
        self.sample_size = self.mesh.size(0)

    def __len__(self):
        return self.sample_size

    # We return idx just so the DataLoader doesn't complain.
    def __getitem__(self, idx):
        return self.mesh[idx], idx


def get_rand_domain_loader(data_shape, sample_size, batch_size):
    dataset = RandomDomainDataset(data_shape, sample_size)
    return DataLoader(dataset, batch_size, shuffle=True)


def get_mesh_domain_loader(data_shape, epsilon):
    dataset = MeshDomainDataset(data_shape, epsilon)
    return DataLoader(dataset, batch_size=len(dataset), shuffle=True)


def get_epsilon_mesh(epsilon, data_shape, device):
    cell_width = torch.sqrt(torch.tensor(2.0, device=device)) * epsilon
    num_cells = int(torch.ceil(1 / cell_width))
    actual_cell_width = 1 / num_cells
    actual_epsilon = actual_cell_width / torch.sqrt(torch.tensor(2.0, device=device))
    num_pixels = data_shape[0] * data_shape[1]
    print(
        f"Creating mesh with {num_cells}^{num_pixels} = {num_cells ** num_pixels} elements."
    )
    mesh = torch.cartesian_prod(
        *[torch.linspace(0, 1, num_cells + 1, device=device)] * num_pixels
    )
    return mesh, actual_epsilon, actual_cell_width
