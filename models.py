import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms


class MLP(nn.Module):
    def __init__(self, dimensions):
        super(MLP, self).__init__()
        self.dimensions = dimensions
        self.layers = []
        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:  # No activation on the last layer
                self.layers.append(nn.ReLU())
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)
    
    def num_parameters(self):
        return sum([layer.weight.numel() + layer.bias.numel() for layer in self.network if isinstance(layer, nn.Linear)])


class ParameterDataset(Dataset):
    def __init__(self, model: MLP):
        super(ParameterDataset, self).__init__()
        self.num_layers = len(model.dimensions) - 1
        self.max_rows = max(model.dimensions[1:])
        self.max_cols = max(model.dimensions[:-1])

        self.max_vals = torch.tensor([self.num_layers - 1, self.max_rows - 1, self.max_cols])
        self.transform = lambda x: x / self.max_vals - 0.5

        # self.num_vals = torch.tensor([self.num_layers, self.max_rows, self.max_cols + 1])
        # self.mean_vals = torch.tensor([(num - 1) / 2 for num in self.num_vals])  # Mean of uniform distribution on {0, ..., num - 1}
        # self.std_vals = torch.tensor([(num ** 2 - 1) / 12 for num in self.num_vals]).sqrt()  # Variance of uniform distribution on {0, ..., num - 1}
        # self.transform = lambda x: (x - self.mean_vals) / self.std_vals
        self.return_transformed = False

        self.params = []
        for layer_num, layer in enumerate(model.network):
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach()
                bias = layer.bias.detach()
                for row in range(weight.size(0)):
                    for col in range(weight.size(1)):
                        x = torch.tensor([layer_num, row, col])
                        y = torch.tensor([weight[row, col]])
                        self.params.append((x, y))
                for row in range(bias.size(0)):
                    x = torch.tensor([layer_num, row, weight.size(1)])
                    y = torch.tensor([bias[row]])
                    self.params.append((x, y))

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        x, y = self.params[idx]
        if self.return_transformed:
            return self.transform(x), y
        return x, y
