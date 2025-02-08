import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Subset

# Load MNIST

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = torchvision.datasets.MNIST(root='./data/delete', train=True, download=True, transform=transform)
# print(trainset.data)
# print(trainset.targets)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# dataset_size = 15
# x_train = torch.randn(dataset_size, 10)
# y_train = torch.randint(0, 10, (dataset_size,))
# train = torch.utils.data.TensorDataset(x_train, y_train)
# print(dir(train))


train = datasets.MNIST(
    root=".data/delete",
    train=True,
    download=True,
)
# train = Subset(train, range(100))
print(type(train.dataset.data))
print(type(train.dataset.targets))