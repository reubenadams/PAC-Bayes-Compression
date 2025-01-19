import torch
from torchvision import datasets, transforms


def get_dataloaders(dataset_name, batch_size, shrink_test_dataset=False):

    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # Tried with transforms.Normalize((0.1307,), (0.3081,)) but this increased max fro norm from 28.0 to 48.9
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if dataset_name == "CIFAR100":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((10, 10)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    if shrink_test_dataset:
        test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader


def get_B(data_loader):
    max_fro_norm = torch.tensor(0.0)
    for data, _ in data_loader:
        fro_norms = torch.linalg.matrix_norm(data, ord='fro')
        max_fro_norm = max(max_fro_norm, fro_norms.max())
    return max_fro_norm
