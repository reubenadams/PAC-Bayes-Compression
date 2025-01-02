import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import wandb

from models import BaseModel, HyperModel
from config import Config


base_config = Config(
    model_path="trained_models/base_mlp.t",
    model_dims=(784, 128, 10),
    model_act="relu",
    train_epochs=5,
    batch_size=64,
    learning_rate=0.001
)

hyper_config = Config(
    model_path="trained_models/hyper_mlp.t",
    model_dims=(3, 32, 256, 1024, 256, 256, 32, 1),
    model_act="sigmoid",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.000001
)
base_model_estimate_path = "trained_models/base_mlp_estimate.t"


wandb.init(project="hypertraining", name="Hyper [3, 256, 256, 256, 256, 1], lr=0.000001")

os.makedirs("trained_models", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=base_config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=base_config.batch_size, shuffle=False)


base_model = BaseModel(base_config.model_dims, base_config.model_act)
base_model_estimate = BaseModel(base_config.model_dims, base_config.model_act)
hyper_model = HyperModel(hyper_config.model_dims, hyper_config.model_act)

print(f"Number of parameters in base model: {base_model.num_parameters}")
print(f"Number of parameters in hyper model: {hyper_model.num_parameters}")


try:
    base_model.load(base_config.model_path)
    print(f"File {base_config.model_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {base_config.model_path} not found. Training model...")
    base_train_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    base_test_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=base_config.learning_rate)
    base_model.train(
        train_loss_fn=base_train_loss_fn,
        test_loss_fn=base_test_loss_fn,
        optimizer=base_optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=base_config.train_epochs,
        log_name="base_train_loss"
        )
    base_model.save(base_config.model_path)

param_dataset = base_model.get_parameter_dataset()
param_dataloader = torch.utils.data.DataLoader(dataset=param_dataset, batch_size=hyper_config.batch_size, shuffle=True)

try:
    hyper_model.load(hyper_config.model_path)
    print(f"File {hyper_config.model_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {hyper_config.model_path} not found. Training model...")
    hyper_train_loss_fn = torch.nn.MSELoss(reduction='mean')
    hyper_test_loss_fn = torch.nn.MSELoss(reduction='sum')
    hyper_optimizer = torch.optim.Adam(hyper_model.parameters(), lr=hyper_config.learning_rate)
    hyper_model.train(
        train_loss_fn=hyper_train_loss_fn,
        test_loss_fn=hyper_test_loss_fn,
        optimizer=hyper_optimizer,
        train_loader=param_dataloader,
        test_loader=param_dataloader,
        num_epochs=hyper_config.train_epochs,
        log_name="hyper_train_loss"
        )
    hyper_model.save(hyper_config.model_path)


try:
    base_model_estimate.load(base_model_estimate_path)
    print(f"File {base_model_estimate_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {base_model_estimate_path} not found. Estimating model...")
    base_model_estimate.load_from_hypermodel(hyper_model)
    base_model_estimate.save(base_model_estimate_path)
    print("Finished saving.")

print(f"Base model accuracy: {base_model.overall_accuracy(test_loader).item()}")
print(f"Base model estimate accuracy: {base_model_estimate.overall_accuracy(test_loader).item()}")