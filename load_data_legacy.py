import os
from typing import Optional, Union
from math import prod

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from mnist1d.data import get_dataset_args, get_dataset, make_dataset
from mnist1d.utils import ObjectView


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
        train = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
        test = torch.load(os.path.join(data_dir, "test.pt"), weights_only=False)
        print(f"Data successfully loaded from {data_dir}.")

    except FileNotFoundError:

        print(f"Data not found in {data_dir}. Downloading (MNIST, CIFAR10) or generating (MNIST1D) data...")
        print(f"{new_input_shape=}, {train_size=}, {test_size=}")
        transform = transforms.Compose(
            [
                transforms.Resize(_new_input_shape),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        if dataset_name == "MNIST":
            print("Downloading MNIST dataset with transform equal to: ", transform)

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

            # Inspect datatype:
            print(f"{train.data.dtype=}")
            if _train_size is not None:
                train = CustomSubset(dataset=train, indices=range(_train_size))
            if _test_size is not None:
                test = CustomSubset(dataset=test, indices=range(_test_size))

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
                train = CustomSubset(dataset=train, indices=range(_train_size))
            if _test_size is not None:
                test = CustomSubset(dataset=test, indices=range(_test_size))

        elif dataset_name == "MNIST1D":
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(data_dir, "mnist1d_data.pkl")
            
            if _train_size is None and _test_size is None:
                args = get_dataset_args()
            elif _train_size is not None and _test_size is not None:
                args = get_dataset_args(as_dict=True)
                args["num_samples"] = _train_size + _test_size
                args["train_split"] = _train_size / (train_size + test_size)
                args = ObjectView(args)
            else:
                raise ValueError("Either both train_size and test_size should be specified or neither.")

            data = make_dataset(args=args)
            # data = get_dataset(args, path=path, download=True)
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


class FakeDataLoader:
    """Fake DataLoader that returns the whole dataset in one batch."""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.dataset = data  # This is a fictional attribute so we can call len(dataloader.dataset)

    def __iter__(self):
        return iter([(self.data, self.targets)])


def get_dataloaders(
    dataset_name,
    batch_size,
    train_size,
    test_size,
    new_input_shape,
    train_dataset=None,
    test_dataset=None,
    data_dir=None,
    use_whole_dataset=False,
    device="cpu",
) -> Union[DataLoader, FakeDataLoader]:

    if train_dataset is None or test_dataset is None or data_dir is None:
        print("At least one of train_dataset or test_dataset is None. Loading datasets...")
        train_dataset, test_dataset, data_dir = get_datasets(
            dataset_name=dataset_name,
            new_input_shape=new_input_shape,
            train_size=train_size,
            test_size=test_size,
        )

    if train_dataset is not None or test_dataset is not None:
        assert (
            (type(train_dataset) == Dataset and type(test_dataset) == Dataset) or
            (type(train_dataset) == CustomDataset and type(test_dataset) == CustomDataset) or
            (type(train_dataset) == CustomSubset and type(test_dataset) == CustomSubset)
        ), f"train and test should be both Dataset, CustomDataset or CustomSubset instances. Got {type(train_dataset)} and {type(test_dataset)}."
    # assert (
        # isinstance(train_dataset, Dataset) and isinstance(test_dataset, Dataset)) or (
        # isinstance(train_dataset, Subset) and isinstance(test_dataset, Subset)
    # ), "train and test should be both Dataset or both Subset instances."

    train_dataset.data = train_dataset.data.to(device)
    train_dataset.targets = train_dataset.targets.to(device)
    test_dataset.data = test_dataset.data.to(device)
    test_dataset.targets = test_dataset.targets.to(device)    

    # if isinstance(train_dataset, Subset):
    #     train_dataset.dataset.data = train_dataset.dataset.data.to(device)
    #     train_dataset.dataset.targets = train_dataset.dataset.targets.to(device)
    #     train_dataset.dataset.data = train_dataset.dataset.data.to(device)
    #     train_dataset.dataset.targets = train_dataset.dataset.targets.to(device)
    # else:
    #     train_dataset.data = train_dataset.data.to(device)
    #     train_dataset.targets = train_dataset.targets.to(device)
    #     test_dataset.data = test_dataset.data.to(device)
    #     test_dataset.targets = test_dataset.targets.to(device)

    if use_whole_dataset:
        return FakeDataLoader(train_dataset.data, train_dataset.targets), FakeDataLoader(test_dataset.data, test_dataset.targets), data_dir
        # if isinstance(train_dataset, Subset):
        #     train_data = train_dataset.dataset.data[train_dataset.indices]
        #     train_targets = train_dataset.dataset.targets[train_dataset.indices]
        #     test_data = test_dataset.dataset.data[test_dataset.indices]
        #     test_targets = test_dataset.dataset.targets[test_dataset.indices]
        #     print("Returning fake dataloaders!")
        #     return FakeDataLoader(train_data, train_targets), FakeDataLoader(test_data, test_targets), data_dir
        # else:
        #     print("Returning fake dataloaders!")
        #     return FakeDataLoader(train_dataset.data, train_dataset.targets), FakeDataLoader(test_dataset.data, test_dataset.targets), data_dir

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    print("Returning real dataloaders!")
    return train_loader, test_loader, data_dir


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


class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
        self.data = dataset.data[indices]
        self.targets = dataset.targets[indices]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.indices)


class RandomDomainDataset(Dataset):

    def __init__(
        self,
        data_shape,
        sample_size,
        dist_name,
        dist_mean=None,
        dist_std=None,
        dist_min=None,
        dist_max=None,
        device="cpu"
    ):
        self.device = device
        self.data_shape = data_shape
        self.num_pixels = prod(data_shape)
        self.sample_size = sample_size

        if dist_name == "uniform":
            assert dist_min is not None and dist_max is not None and dist_mean is None and dist_std is None, "dist_min and dist_max must be provided for uniform distribution, but not dist_mean and dist_std."
            self.dist = torch.distributions.Uniform(dist_min, dist_max)
        elif dist_name == "normal":
            assert dist_mean is not None and dist_std is not None and dist_min is None and dist_max is None, "dist_mean and dist_std must be provided for normal distribution, but not dist_min and dist_max."
            if dist_std <= 0:
                raise ValueError("Standard deviation must be positive for normal distribution.")
            self.dist = torch.distributions.Normal(dist_mean, dist_std)
        else:
            raise ValueError("Invalid distribution name. Use 'uniform' or 'normal'.")

        self.data = self.dist.sample((sample_size, self.num_pixels)).to(device)
        self.targets = torch.arange(sample_size).to(device)  # We return idx as target just so the DataLoader doesn't complain.

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class MeshDomainDataset(Dataset):

    def __init__(self, data_shape, epsilon):
        self.num_pixels = prod(data_shape)
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


def get_rand_domain_dataset_and_loader(
    data_shape,
    use_whole_dataset,
    sample_size,
    batch_size,
    dist_name,
    dist_mean=None,
    dist_std=None,
    dist_min=None,
    dist_max=None,
    device="cpu"
) -> Union[DataLoader, FakeDataLoader]:
    dataset = RandomDomainDataset(
        data_shape=data_shape,
        sample_size=sample_size,
        dist_name=dist_name,
        dist_mean=dist_mean,
        dist_std=dist_std,
        dist_min=dist_min,
        dist_max=dist_max,
        device=device,
    )
    if use_whole_dataset:
        return dataset, FakeDataLoader(dataset.data, dataset.targets)
    return dataset, DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


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


def get_logits_loader(model, dataset: Union[Dataset, RandomDomainDataset], use_whole_dataset, batch_size) -> FakeDataLoader:
    if not use_whole_dataset or batch_size is not None:
        raise NotImplementedError("Logit loaders are not implemented for use_whole_dataset=False, as dataloaders and logit loaders can get out of sink.")
    inputs = dataset.data.to(model.device)
    targets = dataset.targets.to(model.device)
    assert not model.training, "Model must be in eval mode to get logits."
    with torch.no_grad():
        print(f"{inputs.dtype=}")
        print(f"{inputs.device=}")
        print(f"{inputs.shape=}")
        print(inputs)
        logits = model(inputs)
    return FakeDataLoader(logits, targets)


def check_logits_loader(model, data_loader, logits_loader):
    """Check that the logits loader returns the same logits as the model on the data loader."""
    for _ in range(2):
        for (x, y1), (logits, y2) in zip(data_loader, logits_loader):
            x = x.to(model.device)
            logits = logits.to(model.device)
            y1 = y1.to(model.device)
            y2 = y2.to(model.device)
            assert torch.allclose(y1, y2), "Labels do not match in data_loader and logits_loader."
            assert torch.allclose(model(x), logits), "Logits do not match model output."
