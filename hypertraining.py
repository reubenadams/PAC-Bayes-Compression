import os
from copy import deepcopy

import torch
import wandb

from models import BaseMLP, get_reconstructed_accuracy
from config import (
    base_mnist_config,
    hyper_mnist_config_scaled,
    hyper_mnist_config_binary,
)
from load_data import get_dataloaders


to_train = {
    "base": True,
    "hyper_scaled": True,
    "hyper_binary": True,
}


train_loader, test_loader = get_dataloaders(
    base_mnist_config.dataset,
    base_mnist_config.batch_size,
    train_size=100,
    test_size=100,
)


base_model_estimate_scaled_path = "trained_models/base_mlp_estimate_scaled.t"
base_model_estimate_binary_path = "trained_models/base_mlp_estimate_binary.t"


wandb.init(
    project="hypertraining",
    name=f"Base: dims={base_mnist_config.model_dims}, Hyper binary: dims={hyper_mnist_config_binary.model_dims}, act={hyper_mnist_config_binary.model_act} lr={hyper_mnist_config_binary.lr}",
)


base_model = BaseMLP(base_mnist_config.model_dims, base_mnist_config.model_act)
hyper_model_scaled = base_model.get_hyper_model_scaled_input(hyper_mnist_config_scaled)
print(f"hyper config binary dims: {hyper_mnist_config_binary.model_dims}")
hyper_model_binary = base_model.get_hyper_model_binary_input(hyper_mnist_config_binary)
print(f"hyper config binary dims: {hyper_mnist_config_binary.model_dims}")


print(f"Number of parameters in base model: {base_model.num_parameters}")
print(
    f"Number of parameters in hyper model scaled: {hyper_model_scaled.num_parameters}"
)
print(
    f"Number of parameters in hyper model binary: {hyper_model_binary.num_parameters}"
)


try:
    base_model.load(base_mnist_config.model_path)
    print(f"File {base_mnist_config.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["base"]:
        print(f"File {base_mnist_config.model_path} not found. Training model...")
        base_train_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        base_test_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        base_model.train(
            train_loss_fn=base_train_loss_fn,
            test_loss_fn=base_test_loss_fn,
            lr=base_mnist_config.lr,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=base_mnist_config.train_epochs,
            get_test_loss=True,
            get_test_accuracy=True,
            train_loss_name="Base Train Loss",
            test_loss_name="Base Test Loss",
            test_accuracy_name="Base Test Accuracy",
        )
        base_model.save(base_mnist_config.model_dir, base_mnist_config.model_name)


try:
    hyper_model_scaled.load(hyper_mnist_config_scaled.model_path)
    print(f"File {hyper_mnist_config_scaled.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["hyper_scaled"]:
        print(
            f"File {hyper_mnist_config_scaled.model_path} not found. Training model..."
        )
        hyper_train_loss_fn = torch.nn.MSELoss(reduction="mean")
        hyper_test_loss_fn = torch.nn.MSELoss(reduction="sum")
        # hyper_scheduler = torch.optim.lr_scheduler.StepLR(hyper_optimizer, step_size=10, gamma=0.1)
        param_dataset_scaled = base_model.get_parameter_dataset(
            transform=base_model.scale_indices_transform
        )
        param_dataloader_scaled = torch.utils.data.DataLoader(
            dataset=param_dataset_scaled,
            batch_size=hyper_mnist_config_scaled.batch_size,
            shuffle=True,
        )

        def callback(epoch):
            reconstructed_accuracy = get_reconstructed_accuracy(
                base_model,
                hyper_model_scaled,
                base_model.scale_indices_transform,
                test_loader,
            )
            wandb.log({"epoch": epoch, "Rec Acc Scaled": reconstructed_accuracy})

        hyper_model_scaled.train(
            train_loss_fn=hyper_train_loss_fn,
            test_loss_fn=hyper_test_loss_fn,
            lr=hyper_mnist_config_scaled.lr,
            train_loader=param_dataloader_scaled,
            test_loader=param_dataloader_scaled,
            num_epochs=hyper_mnist_config_scaled.train_epochs,
            train_loss_name="hyper_scaled_train_loss",
            callback=callback,
        )
        hyper_model_scaled.save(hyper_mnist_config_scaled.model_dir, hyper_mnist_config_scaled.model_name)


try:
    hyper_model_binary.load(hyper_mnist_config_binary.model_path)
    print(f"File {hyper_mnist_config_binary.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["hyper_binary"]:
        print(
            f"File {hyper_mnist_config_binary.model_path} not found. Training model..."
        )
        hyper_train_loss_fn = torch.nn.MSELoss(reduction="mean")
        hyper_test_loss_fn = torch.nn.MSELoss(reduction="sum")
        param_dataset_binary = base_model.get_parameter_dataset(
            transform=base_model.binary_indices_transform
        )
        param_dataloader_binary = torch.utils.data.DataLoader(
            dataset=param_dataset_binary,
            batch_size=hyper_mnist_config_binary.batch_size,
            shuffle=True,
        )

        def callback(epoch):
            reconstructed_accuracy = get_reconstructed_accuracy(
                base_model,
                hyper_model_binary,
                base_model.binary_indices_transform,
                test_loader,
            )
            wandb.log({"epoch": epoch, "Rec Acc Binary": reconstructed_accuracy})

        hyper_model_binary.train(
            train_loss_fn=hyper_train_loss_fn,
            test_loss_fn=hyper_test_loss_fn,
            lr=hyper_mnist_config_binary.lr,
            train_loader=param_dataloader_binary,
            test_loader=param_dataloader_binary,
            num_epochs=hyper_mnist_config_binary.train_epochs,
            train_loss_name="hyper_binary_train_loss",
            callback=callback,
        )
        hyper_model_binary.save(hyper_mnist_config_binary.model_dir, hyper_mnist_config_binary.model_name)
        reconstructed_accuracy = get_reconstructed_accuracy(
            base_model,
            hyper_model_binary,
            base_model.binary_indices_transform,
            test_loader,
        )
        print(f"Base model estimate scaled, accuracy: {reconstructed_accuracy}")
