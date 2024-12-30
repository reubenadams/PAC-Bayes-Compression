import os

import torch
from torchvision import datasets, transforms

from models import BaseModel, HyperModel


os.makedirs("trained_models", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


base_model_dims = [784, 128, 10]
base_model = BaseModel(base_model_dims)
base_model_estimate = BaseModel(base_model_dims)
base_model_path = "trained_models/base_mlp.t"
base_model_estimate_path = "trained_models/base_estimate_mlp.t"

hyper_model_dims = [3, 128, 1]
hyper_model = HyperModel(hyper_model_dims)
hyper_model_path = "trained_models/hyper_mlp.t"

print(f"Number of parameters in base model: {base_model.num_parameters()}")
print(f"Number of parameters in hyper model: {hyper_model.num_parameters()}")


try:
    base_model.load(base_model_path)
    print(f"File {base_model_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {base_model_path} not found. Training model...")
    base_train_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    base_test_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
    base_num_epochs = 5
    base_model.train(
        train_loss_fn=base_train_loss_fn,
        test_loss_fn=base_test_loss_fn,
        optimizer=base_optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=base_num_epochs
        )
    base_model.save(base_model_path)

param_dataset = base_model.get_parameter_dataset()
param_dataloader = torch.utils.data.DataLoader(dataset=param_dataset, batch_size=64, shuffle=True)

try:
    hyper_model.load(hyper_model_path)
    print(f"File {hyper_model_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {hyper_model_path} not found. Training model...")
    hyper_train_loss_fn = torch.nn.MSELoss(reduction='mean')
    hyper_test_loss_fn = torch.nn.MSELoss(reduction='sum')
    hyper_optimizer = torch.optim.Adam(hyper_model.parameters(), lr=0.001)
    hyper_num_epochs = 5
    hyper_model.train(
        train_loss_fn=hyper_train_loss_fn,
        test_loss_fn=hyper_test_loss_fn,
        optimizer=hyper_optimizer,
        train_loader=param_dataloader,
        test_loader=param_dataloader,
        num_epochs=hyper_num_epochs
        )
    hyper_model.save(hyper_model_path)


try:
    base_model_estimate.load(base_model_estimate_path)
    print(f"File {base_model_estimate_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {base_model_estimate_path} not found. Estimating model...")
    base_model_estimate.load_from_hypermodel(hyper_model)
    print("Finished estimating. Now saving")
    base_model_estimate.save(base_model_estimate_path)
    print("Finished saving.")
