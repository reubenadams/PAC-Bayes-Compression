import torch
from torch.utils.data import Dataset
import torch.nn as nn


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
    
    @property
    def linear_layers(self):
        return (layer for layer in self.network if isinstance(layer, nn.Linear))

    def num_parameters(self):
        return sum([layer.weight.numel() + layer.bias.numel() for layer in self.linear_layers])
    
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
            _, predicted = torch.max(outputs, 1)
            num_correct += (predicted == labels).sum().item()
        return num_correct / len(dataloader.dataset)
    
    def train(self, train_loss_fn, test_loss_fn, optimizer, train_loader, test_loader, num_epochs):
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images = images.view(images.size(0), -1)
                outputs = self(images)
                loss = train_loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            test_loss = self.overall_loss(test_loss_fn, test_loader)
            test_accuracy = self.overall_accuracy(test_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy.item():.4f}')
        print("Training complete.")
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

class HyperModel(MLP):
    def __init__(self, dimensions):
        assert dimensions[0] == 3, "The first dimension of the hypermodel must be 3"
        assert dimensions[-1] == 1, "The last dimension of the hypermodel must be 1"
        super(HyperModel, self).__init__(dimensions)

        self.num_layers = len(self.dimensions) - 1
        self.max_rows = max(self.dimensions[1:])
        self.max_cols = max(self.dimensions[:-1])

        self.max_vals = torch.tensor([self.num_layers - 1, self.max_rows - 1, self.max_cols])
        self.transform = lambda x: x / self.max_vals - 0.5  # Normalize indices to [-0.5, 0.5]
        self.treat_input_as_raw_index = True

    def forward(self, x):
        if self.treat_input_as_raw_index:
            return self.network(self.transform(x))
        return self.network(x)


class BaseModel(MLP):
    def __init__(self, dimensions):
        super(BaseModel, self).__init__(dimensions)

    def get_parameter_dataset(self):
        return ParameterDataset(self)

    def load_from_hypermodel(self, hypermodel: HyperModel):
        """Populate the weights of the base model with the estimated weights from the hypermodel"""
        assert hypermodel.treat_input_as_raw_index == True
        for layer_num, layer in enumerate(self.linear_layers):
            print(f"Estimating layer {layer_num}...")
            weight = torch.zeros_like(layer.weight)
            bias = torch.zeros_like(layer.bias)
            for row in range(weight.size(0)):
                print(f"\tEstimating row {row}...")
                for col in range(weight.size(1)):
                    print(f"\t\tEstimating col {col}...")
                    x = torch.tensor([layer_num, row, col])
                    weight[row, col] = hypermodel(x)
            for row in range(bias.size(0)):
                print(f"\tEstimating bias row {row}...")
                x = torch.tensor([layer_num, row, weight.size(1)])
                bias[row] = hypermodel(x)
            layer.weight.data = weight
            layer.bias.data = bias


class ParameterDataset(Dataset):
    def __init__(self, model: MLP):
        super(ParameterDataset, self).__init__()
        self.params = []
        for layer_num, layer in enumerate(model.linear_layers):
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
        return self.params[idx]
