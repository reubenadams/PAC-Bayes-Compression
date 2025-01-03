import os
from copy import deepcopy

import torch
from torchvision import datasets, transforms
import wandb

from models import MLP
from config import Config


to_train = {
    "base": False,
    "hyper_scaled": False,
    "hyper_binary": True,
}


base_config = Config(
    model_path="trained_models/base_mlp.t",
    model_dims=[784, 128, 10],
    model_act="relu",
    train_epochs=10,
    batch_size=64,
    learning_rate=0.001
)

hyper_config_scaled = Config(
    model_path="trained_models/hyper_mlp_scaled.t",
    model_dims=[3, 128, 128, 1],
    model_act="relu",
    train_epochs=30,
    batch_size=64,
    learning_rate=0.01
)

hyper_config_binary = Config(
    model_path="trained_models/hyper_mlp_binary.t",
    model_dims=[3, 512, 128, 1],
    model_act="relu",
    train_epochs=100,
    batch_size=64,
    learning_rate=0.001
)

base_model_estimate_scaled_path = "trained_models/base_mlp_estimate_scaled.t"
base_model_estimate_binary_path = "trained_models/base_mlp_estimate_binary.t"


wandb.init(project="hypertraining", name=f"Base: dims={base_config.model_dims}, Hyper binary: dims={hyper_config_binary.model_dims}, act={hyper_config_binary.model_act} lr={hyper_config_binary.learning_rate}")

os.makedirs("trained_models", exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=base_config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=base_config.batch_size, shuffle=False)


base_model = MLP(base_config.model_dims, base_config.model_act)
base_model_estimate_scaled = deepcopy(base_model)
base_model_estimate_binary = deepcopy(base_model)
hyper_model_scaled = base_model.get_hyper_model_scaled_input(hyper_config_scaled)
print(f"hyper config binary dims: {hyper_config_binary.model_dims}")
hyper_model_binary = base_model.get_hyper_model_binary_input(hyper_config_binary)
print(f"hyper config binary dims: {hyper_config_binary.model_dims}")


print(f"Number of parameters in base model: {base_model.num_parameters}")
print(f"Number of parameters in hyper model scaled: {hyper_model_scaled.num_parameters}")
print(f"Number of parameters in hyper model binary: {hyper_model_binary.num_parameters}")



try:
    base_model.load(base_config.model_path)
    print(f"File {base_config.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["base"]:
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
            log_name="base_train_loss",
            get_accuracy=True
            )
        base_model.save(base_config.model_path)


try:
    hyper_model_scaled.load(hyper_config_scaled.model_path)
    print(f"File {hyper_config_scaled.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["hyper_scaled"]:
        print(f"File {hyper_config_scaled.model_path} not found. Training model...")
        hyper_train_loss_fn = torch.nn.MSELoss(reduction='mean')
        hyper_test_loss_fn = torch.nn.MSELoss(reduction='sum')
        hyper_optimizer = torch.optim.Adam(hyper_model_scaled.parameters(), lr=hyper_config_scaled.learning_rate)
        param_dataset_scaled = base_model.get_parameter_dataset(transform=base_model.scale_indices_transform)
        param_dataloader_scaled = torch.utils.data.DataLoader(dataset=param_dataset_scaled, batch_size=hyper_config_scaled.batch_size, shuffle=True)
        hyper_model_scaled.train(
            train_loss_fn=hyper_train_loss_fn,
            test_loss_fn=hyper_test_loss_fn,
            optimizer=hyper_optimizer,
            train_loader=param_dataloader_scaled,
            test_loader=param_dataloader_scaled,
            num_epochs=hyper_config_scaled.train_epochs,
            log_name="hyper_scaled_train_loss"
            )
        hyper_model_scaled.save(hyper_config_scaled.model_path)


try:
    hyper_model_binary.load(hyper_config_binary.model_path)
    print(f"File {hyper_config_binary.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["hyper_binary"]:
        print(f"File {hyper_config_binary.model_path} not found. Training model...")
        hyper_train_loss_fn = torch.nn.MSELoss(reduction='mean')
        hyper_test_loss_fn = torch.nn.MSELoss(reduction='sum')
        hyper_optimizer = torch.optim.Adam(hyper_model_binary.parameters(), lr=hyper_config_binary.learning_rate)
        param_dataset_binary = base_model.get_parameter_dataset(transform=base_model.binary_indices_transform)
        param_dataloader_binary = torch.utils.data.DataLoader(dataset=param_dataset_binary, batch_size=hyper_config_binary.batch_size, shuffle=True)
        hyper_model_binary.train(
            train_loss_fn=hyper_train_loss_fn,
            test_loss_fn=hyper_test_loss_fn,
            optimizer=hyper_optimizer,
            train_loader=param_dataloader_binary,
            test_loader=param_dataloader_binary,
            num_epochs=hyper_config_binary.train_epochs,
            log_name="hyper_binary_train_loss"
            )
        hyper_model_binary.save(hyper_config_binary.model_path)


if to_train["hyper_scaled"]:
    try:
        base_model_estimate_scaled.load(base_model_estimate_scaled_path)
        print(f"File {base_model_estimate_scaled_path} found. Loading model...")
    except FileNotFoundError:
        print(f"File {base_model_estimate_scaled_path} not found. Estimating model...")
        base_model_estimate_scaled.load_from_hyper_model(hyper_model_scaled, transform=base_model.scale_indices_transform)
        base_model_estimate_scaled.save(base_model_estimate_scaled_path)
        print("Finished saving.")


if to_train["hyper_binary"]:
    try:
        base_model_estimate_binary.load(base_model_estimate_binary_path)
        print(f"File {base_model_estimate_binary_path} found. Loading model...")
    except FileNotFoundError:
        print(f"File {base_model_estimate_binary_path} not found. Estimating model...")
        base_model_estimate_binary.load_from_hyper_model(hyper_model_binary, transform=base_model.binary_indices_transform)
        base_model_estimate_binary.save(base_model_estimate_binary_path)
        print("Finished saving.")


print(f"Base model accuracy: {base_model.overall_accuracy(test_loader).item()}")
if to_train["hyper_scaled"]:
    print(f"Base model estimate scaled, accuracy: {base_model_estimate_scaled.overall_accuracy(test_loader).item()}")
if to_train["hyper_binary"]:
    print(f"Base model estimate binary, accuracy: {base_model_estimate_binary.overall_accuracy(test_loader).item()}")
