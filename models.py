import torch
from torch.utils.data import Dataset
import torch.nn as nn
import wandb

from copy import deepcopy

from config import Config


class MLP(nn.Module):
    def __init__(self, dimensions, activation):
        super(MLP, self).__init__()
        self.dimensions = dimensions
        self.activation = self.get_act(activation)
        self.layers = []
        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:  # No activation on the last layer
                self.layers.append(self.activation())
        self.network = nn.Sequential(*self.layers)

        self.num_layers = len(dimensions) - 1
        self.max_rows = max(dimensions[1:])
        self.max_cols = max(dimensions[:-1])
        self.max_indices = (self.num_layers - 1, self.max_rows - 1, self.max_cols)  # Extra col is for bias
        self.bit_lengths = (self.num_layers.bit_length(), self.max_rows.bit_length(), self.max_cols.bit_length())
        self.num_parameters = sum([layer.weight.numel() + layer.bias.numel() for layer in self.linear_layers])

    def forward(self, x):
        return self.network(x)
    
    @property
    def linear_layers(self):
        return (layer for layer in self.network if isinstance(layer, nn.Linear))

    @staticmethod
    def scale_indices(indices, max_indices):
        return torch.tensor(indices, dtype=torch.float) / torch.tensor(max_indices, dtype=torch.float) - 0.5

    @property
    def scale_indices_transform(self):
        max_indices = [max(1, idx) for idx in self.max_indices]  # To avoid division by zero if there is only one layer/row/col
        return lambda indices: self.scale_indices(indices, max_indices)

    @staticmethod
    def binary_indices(indices, bit_lengths):
        binary_string = ''.join([to_padded_binary(idx, num_bits) for idx, num_bits in zip(indices, bit_lengths)])
        return torch.tensor([int(d) for d in binary_string], dtype=torch.float) - 0.5

    @property
    def binary_indices_transform(self):
        return lambda indices: self.binary_indices(indices, self.bit_lengths)

    def get_parameter_dataset(self, transform=None):
        return ParameterDataset(self, transform)

    def overall_loss(self, loss_fn, dataloader):
        assert loss_fn.reduction == 'sum'
        total_loss = torch.tensor(0.)
        for x, labels in dataloader:
            x = x.view(x.size(0), -1)
            outputs = self(x)
            total_loss += loss_fn(outputs, labels)
        return total_loss / len(dataloader.dataset)

    def overall_accuracy(self, dataloader):
        num_correct = torch.tensor(0.)
        for x, labels in dataloader:
            x = x.view(x.size(0), -1)
            outputs = self(x)
            _, predicted = torch.max(outputs, -1)
            num_correct += (predicted == labels).sum().item()
        return num_correct / len(dataloader.dataset)
    
    def train(self, train_loss_fn, test_loss_fn, optimizer, train_loader, test_loader, num_epochs, scheduler=None, log_name=None, get_accuracy=False, callback=None):
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images = images.view(images.size(0), -1)
                outputs = self(images)
                loss = train_loss_fn(outputs, labels)
                if log_name:
                    wandb.log({log_name: loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()
            test_loss = self.overall_loss(test_loss_fn, test_loader)
            if get_accuracy:
                test_accuracy = self.overall_accuracy(test_loader)
                print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss.item()}, Test Accuracy: {test_accuracy.item():.4f}')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss.item()}')
            if epoch % 10 == 0 and callback:
                callback(epoch)
        print("Training complete.")
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def load_from_hyper_model(self, hyper_model, transform=None):
        """Populate the weights of the base model with the estimated weights from the hyper_model"""
        with torch.no_grad():
            for layer_num, layer in enumerate(self.linear_layers):
                for row in range(layer.weight.size(0)):
                    for col in range(layer.weight.size(1) + 1):

                        x = [layer_num, row, col]

                        if transform:
                            x = transform(x)
                        param_hat = hyper_model(x)

                        if col < layer.weight.size(1):  # Weight
                            layer.weight[row, col] = param_hat
                        else:  # Bias
                            layer.bias[row] = param_hat

    def get_hyper_model_scaled_input(self, hyper_config: Config):
        assert hyper_config.model_dims[0] == 3, "The first dimension of the hyper_model must be 3"
        assert hyper_config.model_dims[-1] == 1, "The last dimension of the hyper_model must be 1"
        return HyperModel(hyper_config.model_dims, hyper_config.model_act, transform=self.scale_indices_transform)

    def get_hyper_model_binary_input(self, hyper_config: Config):
        if hyper_config.model_dims[0] != sum(self.bit_lengths):
            print(f"Changing dimensions from {hyper_config.model_dims[0]} to {sum(self.bit_lengths)}")
            hyper_config.model_dims[0] = sum(self.bit_lengths)
        assert hyper_config.model_dims[-1] == 1, "The last dimension of the hyper_model must be 1"
        return HyperModel(hyper_config.model_dims, hyper_config.model_act, transform=self.binary_indices_transform)

    @staticmethod
    def get_act(act: str):
        if act == "relu":
            return nn.ReLU
        elif act == "sigmoid":
            return nn.Sigmoid
        elif act == "tanh":
            return nn.Tanh
        elif act == "leaky_relu":
            return nn.LeakyReLU
        else:
            raise ValueError("Invalid activation function")


class HyperModel(MLP):
    def __init__(self, dimensions, activation, transform=None, transform_input=False):
        super(HyperModel, self).__init__(dimensions, activation)
        self.transform = transform
        self.transform_input = transform_input

    def forward(self, x):
        if self.transform_input:
            x = self.transform(x)
        return self.network(x).view(-1)


class ParameterDataset(Dataset):
    def __init__(self, model: MLP, transform=None):
        super(ParameterDataset, self).__init__()
        self.params = []
        with torch.no_grad():
            for layer_num, layer in enumerate(model.linear_layers):
                weight = layer.weight.detach()
                bias = layer.bias.detach()
                for row in range(weight.size(0)):
                    for col in range(weight.size(1) + 1):
                        x = [layer_num, row, col]
                        if transform:
                            x = transform(x)
                        if col < weight.size(1):  # Weight
                            y = weight[row, col]
                        else:  # Bias
                            y = bias[row]
                        self.params.append((x, y))

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx]


def to_padded_binary(n, b):
    return format(n, f'0{b}b')


def get_reconstructed_accuracy(base_model, hyper_model, transform, dataloader):
    base_model_estimate = deepcopy(base_model)
    base_model_estimate.load_from_hyper_model(hyper_model, transform=transform)
    return base_model_estimate.overall_accuracy(dataloader).item()
