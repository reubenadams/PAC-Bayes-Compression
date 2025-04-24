import os
import pickle
from typing import Optional, Union
from math import prod

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from mnist1d.data import get_dataset_args, make_dataset
from mnist1d.utils import ObjectView



class CustomDataset(Dataset):
    """
    Custom dataset that actually stores the data rather than just indices.
    This allows for direct access to the data and targets.
    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target
    
    def __len__(self):
        return len(self.data)


# TODO: Don't get rid of this! While most of the time you only use data and log_probs, in get_empirical_l2_bound you use logits.
class LogitDataset(Dataset):
    def __init__(self, data, targets, logits, log_probs, probs):
        self.data = data
        self.targets = targets
        self.logits = logits
        self.log_probs = log_probs
        self.probs = probs
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.logits[index], self.log_probs[index], self.probs[index]
    
    def __len__(self):
        return len(self.data)


class RandomDomainDataset(Dataset):

    def __init__(
        self,
        data_shape: tuple[int],
        sample_size: int,
        dist_min: float,
        dist_max: float,
        device: str = "cpu",
    ):
        self.device = device
        self.data_shape = data_shape
        self.num_pixels = prod(data_shape)
        self.sample_size = sample_size

        self.dist = torch.distributions.Uniform(dist_min, dist_max)
        self.data = self.dist.sample((sample_size, self.num_pixels)).to(device)
        self.targets = torch.arange(sample_size).to(device)  # We return idx as target just so the DataLoader doesn't complain.

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_logit_dataset(model, dataset: Union[Dataset, CustomDataset, RandomDomainDataset]) -> LogitDataset:
    
    assert not model.training, "Model must be in eval mode to get logits."
    
    # Create a DataLoader to feed the dataset to the model
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    data_list = []
    targets_list = []
    logits_list = []
    log_probs_list = []
    probs_list = []
    with torch.no_grad():
        for x, targets in dataloader:
            x = x.view(x.size(0), -1).to(model.device)
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            data_list.append(x.cpu())
            targets_list.append(targets.cpu())
            logits_list.append(logits.cpu())
            log_probs_list.append(log_probs.cpu())
            probs_list.append(probs.cpu())

    data = torch.cat(data_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    log_probs = torch.cat(log_probs_list, dim=0)
    probs = torch.cat(probs_list, dim=0)
    return LogitDataset(
        data=data,
        targets=targets,
        logits=logits,
        log_probs=log_probs,
        probs=probs,
    )
    

# TODO: This is no longer needed because we can just pass batch_size=len(dataset) to the DataLoader.
# class FakeDataLoader:
#     """Fake DataLoader that returns the whole dataset in one batch."""
#     def __init__(self, data, targets):
#         self.data = data
#         self.targets = targets
#         self.dataset = data  # This is a fictional attribute so we can call len(dataloader.dataset)

#     def __iter__(self):
#         return iter([(self.data, self.targets)])


def get_rand_domain_dataset_and_loader(
    use_whole_dataset: bool,
    batch_size: int,
    data_shape: tuple[int],
    sample_size: int,
    dist_min: float,
    dist_max: float,
    device: str = "cpu",
    ) -> DataLoader:

    dataset = RandomDomainDataset(
        data_shape=data_shape,
        sample_size=sample_size,
        dist_min=dist_min,
        dist_max=dist_max,
        device=device,
    )

    batch_size = update_batch_size(use_whole_dataset, batch_size, dataset)
    return dataset, DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


def get_logit_loader(
        model,
        dataset: Union[Dataset, CustomDataset, RandomDomainDataset],
        use_whole_dataset: bool,
        batch_size: Optional[int],
    ) -> DataLoader:
    batch_size = update_batch_size(use_whole_dataset, batch_size, dataset)
    dist_datset = get_logit_dataset(model, dataset)
    return DataLoader(dist_datset, batch_size=batch_size, shuffle=True)


# def get_logits_loader(model, dataset: Union[Dataset, CustomDataset, RandomDomainDataset], use_whole_dataset, batch_size) -> Union[DataLoader, FakeDataLoader]:
    
#     assert not model.training, "Model must be in eval mode to get logits."
    
#     if use_whole_dataset and batch_size is None:
#         x = dataset.data.to(model.device)
#         with torch.no_grad():
#             x = x.view(x.size(0), -1)
#             logits = model(x)
#         return FakeDataLoader(x, logits)
    
#     if not use_whole_dataset and batch_size is not None:
#         # Create a DataLoader for the dataset
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         logits_list = []
#         with torch.no_grad():
#             for x, _ in dataloader:
#                 x = x.view(x.size(0), -1).to(model.device)
#                 batch_logits = model(x)
#                 logits_list.append(batch_logits.cpu())
#         logits = torch.cat(logits_list, dim=0)
#         return DataLoader(x, logits)

#     else:
#         raise ValueError("Either use_whole_dataset must be True or batch_size must be specified.")


def get_datasets(
        dataset_name: str,
        new_input_shape: Optional[tuple[int]],
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ) -> tuple[CustomDataset, CustomDataset, str]:
    if dataset_name == "MNIST":
        return get_mnist_datasets(new_input_shape=new_input_shape, train_size=train_size, test_size=test_size)
    elif dataset_name == "MNIST1D":
        return get_mnist1d_datasets(train_size=train_size, test_size=test_size)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported. Please choose 'MNIST' or 'MNIST1D'.")


def get_mnist_datasets(
        new_input_shape: tuple[int, int] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None
    ) -> tuple[CustomDataset, CustomDataset, str]:
    """
    Downloads MNIST dataset, resizes and normalizes images, takes subsets of specified sizes,
    saves the datasets, and returns them. If saved datasets already exist, loads and returns them.
    
    Args:
        new_input_shape (tuple): The shape to which images should be resized (height, width)
        train_size (int): Size of the train dataset subset (if None, use the full dataset)
        test_size (int): Size of the test dataset subset (if None, use the full dataset)
    
    Returns:
        tuple: (train_dataset, test_dataset) as CustomDataset objects with direct data access
    """

    if new_input_shape is None:
        new_input_shape = (28, 28)

    # Create a directory to save processed datasets
    data_dir = os.path.join('data', 'MNIST_processed')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate a filename based on the parameters
    data_filename = f"MNIST_{new_input_shape[0]}x{new_input_shape[1]}_train{train_size}_test{test_size}.pkl"
    data_filepath = os.path.join(data_dir, data_filename)
    
    # Check if the file already exists
    if os.path.exists(data_filepath):
        print(f"Loading pre-processed MNIST data from {data_filepath}")
        with open(data_filepath, 'rb') as f:
            train_dataset, test_dataset = pickle.load(f)
        print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
        print(f"Train data stats: min={train_dataset.data.min().item()}, max={train_dataset.data.max().item()}, mean={train_dataset.data.mean().item()}")
        print(f"Test data stats: min={test_dataset.data.min().item()}, max={test_dataset.data.max().item()}, mean={test_dataset.data.mean().item()}")
        return train_dataset, test_dataset, data_filepath
    
    # Check if MNIST has been downloaded already
    mnist_exists = os.path.exists('./data/MNIST/processed/training.pt')
    
    if not mnist_exists:
        print(f"Downloading MNIST data...")
    else:
        print(f"Processing MNIST data to {new_input_shape} with train_size={train_size}, test_size={test_size}...")
    
    # Define transformations for initial loading
    # Note: We'll handle resizing manually after loading to ensure we have direct tensor access
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download the dataset
    train_dataset_full = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset_full = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Load all data into memory
    train_loader = DataLoader(train_dataset_full, batch_size=len(train_dataset_full))
    test_loader = DataLoader(test_dataset_full, batch_size=len(test_dataset_full))
    
    # Get all data in one batch
    train_data, train_targets = next(iter(train_loader))
    test_data, test_targets = next(iter(test_loader))
    
    # Apply resizing
    resize_transform = transforms.Resize(new_input_shape)
    train_data_resized = torch.stack([resize_transform(img.unsqueeze(0)).squeeze(0) for img in train_data])
    test_data_resized = torch.stack([resize_transform(img.unsqueeze(0)).squeeze(0) for img in test_data])
    
    # Create subsets if sizes are specified
    if train_size is not None:
        assert train_size <= len(train_data_resized), f"Requested train_size {train_size} exceeds available data {len(train_data_resized)}"
        # Create random indices without replacement
        train_indices = torch.randperm(len(train_data_resized))[:train_size]
        train_data_subset = train_data_resized[train_indices]
        train_targets_subset = train_targets[train_indices]
    else:
        train_data_subset = train_data_resized
        train_targets_subset = train_targets
    
    if test_size is not None:
        assert test_size <= len(test_data_resized), f"Requested test_size {test_size} exceeds available data {len(test_data_resized)}"
        test_indices = torch.randperm(len(test_data_resized))[:test_size]
        test_data_subset = test_data_resized[test_indices]
        test_targets_subset = test_targets[test_indices]
    else:
        test_data_subset = test_data_resized
        test_targets_subset = test_targets
    
    # Create custom datasets with direct data access
    train_dataset = CustomDataset(train_data_subset, train_targets_subset)
    test_dataset = CustomDataset(test_data_subset, test_targets_subset)
    
    # Save the processed datasets
    with open(data_filepath, 'wb') as f:
        pickle.dump((train_dataset, test_dataset), f)
    
    print(f"Processed MNIST datasets saved to {data_filepath}")
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    print(f"Train data stats: min={train_dataset.data.min().item()}, max={train_dataset.data.max().item()}, mean={train_dataset.data.mean().item()}")
    print(f"Test data stats: min={test_dataset.data.min().item()}, max={test_dataset.data.max().item()}, mean={test_dataset.data.mean().item()}")
    return train_dataset, test_dataset, data_filepath


def get_mnist1d_datasets(
        train_size: Optional[int] = None,
        test_size: Optional[int] = None
    ) -> tuple[CustomDataset, CustomDataset, str]:
    """
    Generates MNIST1D dataset, takes subsets of specified sizes, saves the datasets, and
    returns them. If saved datasets already exist, loads and returns them.
    
    Args:
        train_size (int): Size of the train dataset subset (if None, use the full dataset)
        test_size (int): Size of the test dataset subset (if None, use the full dataset)
    
    Returns:
        tuple: (train_dataset, test_dataset) as CustomDataset objects with direct data access
    """
    # Create a directory to save processed datasets
    data_dir = os.path.join('data', 'MNIST1D_processed')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate a filename based on the parameters
    data_filename = f"MNIST1D_train{train_size}_test{test_size}.pkl"
    data_filepath = os.path.join(data_dir, data_filename)
    
    # Check if the file already exists
    if os.path.exists(data_filepath):
        print(f"Loading pre-processed MNIST1D data from {data_filepath}")
        with open(data_filepath, 'rb') as f:
            train_dataset, test_dataset = pickle.load(f)
        return train_dataset, test_dataset, data_filepath
    
    # Check if MNIST1D has been generated already
    mnist1d_exists = os.path.exists('./data/MNIST1D/processed/training.pt')
    
    if not mnist1d_exists:
        print(f"Generating MNIST1D data...")
    else:
        print(f"Processing MNIST1D data to train_size={train_size}, test_size={test_size}...")
    
    if train_size is None and test_size is None:
        args = get_dataset_args()
    elif train_size is not None and test_size is not None:
        args = get_dataset_args(as_dict=True)
        args["num_samples"] = train_size + test_size
        args["train_split"] = train_size / (train_size + test_size)
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

    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)
    
    # Save the processed datasets
    with open(data_filepath, 'wb') as f:
        pickle.dump((train_dataset, test_dataset), f)
    
    print(f"Processed MNIST1D datasets saved to {data_filepath}")
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset, data_filepath


def get_dataloaders(
    dataset_name,
    train_size: Optional[int],
    test_size: Optional[int],
    new_input_shape: Optional[tuple[int, int]],
    train_dataset: Optional[CustomDataset] = None,
    test_dataset: Optional[CustomDataset] = None,
    data_filepath: Optional[str] = None,
    use_whole_dataset: bool = False,
    batch_size: int = None,
    device: str = "cpu",
    ) -> tuple[DataLoader, DataLoader, str]:

    if train_dataset is None or test_dataset is None or data_filepath is None:
        print("At least one of train_dataset or test_dataset is None. Loading datasets...")
        train_dataset, test_dataset, data_filepath = get_datasets(
            dataset_name=dataset_name,
            new_input_shape=new_input_shape,
            train_size=train_size,
            test_size=test_size,
        )

    train_dataset.data = train_dataset.data.to(device)
    train_dataset.targets = train_dataset.targets.to(device)
    test_dataset.data = test_dataset.data.to(device)
    test_dataset.targets = test_dataset.targets.to(device)    

    train_batch_size = update_batch_size(use_whole_dataset, batch_size, train_dataset)
    test_batch_size = update_batch_size(use_whole_dataset, batch_size, test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader, data_filepath


def get_max_l2_norm_data(dataloader: DataLoader):
    max_l2_norm = torch.tensor(0.0)
    for x, _ in dataloader:
        x = x.view(x.size(0), -1)
        l2_norms = torch.linalg.norm(x, ord=2, dim=1)
        max_l2_norm = max(max_l2_norm, l2_norms.max())
    return max_l2_norm


def update_batch_size(
        use_whole_dataset: bool,
        batch_size: Optional[int],
        dataset: Dataset,
    ) -> int:
    """Update the batch size to len(dataset) if use_whole_dataset is True. Raise error if incompatible arguments."""
    if use_whole_dataset:
        if batch_size is None:
            return len(dataset)
        else:
            raise ValueError("batch_size must be None when use_whole_dataset is True.")
    else:
        if batch_size is None:
            raise ValueError("batch_size must be specified when use_whole_dataset is False.")
        return batch_size


if __name__ == "__main__":
    mlp = lambda x: x  # Dummy model for testing
    mlp.training = False  # Set to eval mode
    mlp.device = "cpu"  # Dummy device for testing
    train_loader, test_loader, data_filepath = get_dataloaders(
        dataset_name="MNIST1D",
        train_size=100,
        test_size=100,
        new_input_shape=(28, 28),
        use_whole_dataset=False,
        batch_size=32,
    )
    logit_loader = get_logit_loader(mlp, train_loader.dataset, batch_size=32)
    for data, targets, logits, log_probs, probs in logit_loader:
        print(f"Data shape: {data.shape}, Targets shape: {targets.shape}, Logits shape: {logits.shape}, Log_probs shape: {log_probs.shape}, Probs shape: {probs.shape}")
        break  # Just to test the first batch
