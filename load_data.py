import os
from typing import Optional

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F

from mnist1d.data import get_dataset_args, get_dataset


dataset_shapes = {
    "MNIST": (28, 28),
    "CIFAR10": (32, 32),
    "MNIST1D": (40,),
}


def get_datasets(
        dataset_name: str,
        new_input_shape: Optional[tuple[int]] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ):

    if new_input_shape is None:
        new_input_shape = "full"
    if train_size is None:
        train_size = "full"
    if test_size is None:   
        test_size = "full"

    valid_datasets = ["MNIST", "CIFAR10", "MNIST1D"]
    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name: {dataset_name} should be one of {valid_datasets}.")

    if new_input_shape == dataset_shapes[dataset_name]:  # Avoids having both "full" and (28, 28) for MNIST, for example.
        new_input_shape = "full"

    _new_input_shape = new_input_shape if isinstance(new_input_shape, tuple) else dataset_shapes[dataset_name]
    _train_size = train_size if isinstance(train_size, int) else None
    _test_size = test_size if isinstance(test_size, int) else None

    data_dir = os.path.join("./data", dataset_name, f"shape_{new_input_shape}", f"train_size_{train_size}_test_size_{test_size}")

    if new_input_shape is not None:
        if dataset_name == "MNIST":
            if _new_input_shape[0] > 28 or _new_input_shape[1] > 28:
                raise ValueError(f"New MNIST size {_new_input_shape} should not be larger than original size 28x28.")
        elif dataset_name == "CIFAR10":
            if _new_input_shape[0] > 32 or _new_input_shape[1] > 32:
                raise ValueError(f"New CIFAR10 size {_new_input_shape} should not be larger than original size 32x32.")
        elif dataset_name == "MNIST1D":
            if _new_input_shape != (40,):
                raise ValueError(f"MNIST1D does not support resizing.")

    try:
        print(f"Loading data from {data_dir}...")
        train = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
        test = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)

    except FileNotFoundError:

        print(f"{new_input_shape=}, {train_size=}, {test_size=}")
        transform = transforms.Compose(
            [
                transforms.Resize(_new_input_shape),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        if dataset_name == "MNIST":

            train = datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=transform,
            )
            test = datasets.MNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=transform,
            )

            if _train_size is not None:
                train = Subset(dataset=train, indices=range(_train_size))
            if _test_size is not None:
                test = Subset(dataset=test, indices=range(_test_size))

        elif dataset_name == "CIFAR10":

            train = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=transform,
            )
            test = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=transform,
            )

            if _train_size is not None:
                train = Subset(dataset=train, indices=range(_train_size))
            if _test_size is not None:
                test = Subset(dataset=test, indices=range(_test_size))

        elif dataset_name == "MNIST1D":
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(data_dir, "mnist1d_data.pkl")
            args = get_dataset_args()
            data = get_dataset(args, path=path, download=True)
            x_train = torch.tensor(data["x"], dtype=torch.float32)
            x_test = torch.tensor(data["x_test"], dtype=torch.float32)
            x_train.clip_(-4.0, 4.0)
            x_test.clip_(-4.0, 4.0)
            y_train, y_test = torch.tensor(data["y"]), torch.tensor(data["y_test"])

            if _train_size is not None:
                x_train, y_train = x_train[:_train_size], y_train[:_train_size]
            if _test_size is not None:
                x_test, y_test = x_test[:_test_size], y_test[:_test_size]

            train = CustomDataset(x_train, y_train)
            test = CustomDataset(x_test, y_test)

        os.makedirs(data_dir, exist_ok=True)
        torch.save(train, os.path.join(data_dir, "train.pt"))
        torch.save(test, os.path.join(data_dir, "test.pt"))

    return train, test, data_dir


def get_dataloaders(
    dataset_name,
    batch_size,
    train_size,
    test_size,
    new_input_shape,
    use_whole_dataset=False,
    device="cpu",
):

    train, test, data_dir = get_datasets(
        dataset_name=dataset_name,
        new_input_shape=new_input_shape,
        train_size=train_size,
        test_size=test_size,
    )

    assert (
        isinstance(train, Dataset) and isinstance(test, Dataset)) or (
        isinstance(train, Subset) and isinstance(test, Subset)
    ), "train and test should be both Dataset or both Subset instances."

    if isinstance(train, Subset):
        train.dataset.data = train.dataset.data.to(device)
        train.dataset.targets = train.dataset.targets.to(device)
        train.dataset.data = train.dataset.data.to(device)
        train.dataset.targets = train.dataset.targets.to(device)
    else:
        train.data = train.data.to(device)
        train.targets = train.targets.to(device)
        test.data = test.data.to(device)
        test.targets = test.targets.to(device)


    if use_whole_dataset:
        if isinstance(train, Subset):
            train_data = train.dataset.data[train.indices]
            train_targets = train.dataset.targets[train.indices]
            test_data = test.dataset.data[test.indices]
            test_targets = test.dataset.targets[test.indices]
            return FakeDataLoader(train_data, train_targets), FakeDataLoader(test_data, test_targets), data_dir
        else:
            return FakeDataLoader(train.data, train.targets), FakeDataLoader(test.data, test.targets), data_dir

    train_loader = DataLoader(train, batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size, shuffle=False)

    return train_loader, test_loader, data_dir


class FakeDataLoader:
    """Fake DataLoader that returns the whole dataset in one batch."""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.dataset = data  # This is a fictional attribute so we can call len(dataloader.dataset)

    def __iter__(self):
        return iter([(self.data, self.targets)])


def get_max_l2_norm_data(dataloader: DataLoader):
    max_l2_norm = torch.tensor(0.0)
    for x, _ in dataloader:
        x = x.view(x.size(0), -1)
        l2_norms = torch.linalg.norm(x, ord=2, dim=1)
        max_l2_norm = max(max_l2_norm, l2_norms.max())
    return max_l2_norm


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


def get_logits_dataloader(model, data_loader, batch_size, use_whole_dataset, device):
    model.to(device)
    inputs = []
    logits = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            print(f"{data.dtype=}")
            print(f"{model.network_modules[0].weight.dtype=}")
            inputs.append(data)
            logits.append(F.log_softmax(model(data), dim=-1))
    inputs = torch.cat(inputs)
    logits = torch.cat(logits)

    if use_whole_dataset:
        return FakeDataLoader(inputs, logits)
    
    logits_dataset = CustomDataset(inputs, logits)
    logits_loader = DataLoader(logits_dataset, batch_size=batch_size, shuffle=True)
    return logits_loader
