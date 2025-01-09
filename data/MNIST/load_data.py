import torch
from torchvision import datasets, transforms

from config import base_config


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # Tried with transforms.Normalize((0.1307,), (0.3081,)) but this increased max fro norm from 28.0 to 48.9
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=base_config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=base_config.batch_size, shuffle=False)


def get_B(data_loader):
    max_fro_norm = torch.tensor(0.0)
    for data, _ in data_loader:
        fro_norms = torch.linalg.matrix_norm(data, ord='fro')
        max_fro_norm = max(max_fro_norm, fro_norms.max())
    return max_fro_norm
