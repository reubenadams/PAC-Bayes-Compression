import os
from copy import deepcopy

import torch
import wandb

from models import BaseMLP, get_reconstructed_accuracy
from config import BaseTrainConfig, ExperimentConfig
from load_data import get_dataloaders


to_train = {
    "base": True,
    "hyper_scaled": True,
    "hyper_binary": True,
}


base_train_config = BaseTrainConfig(
    get_full_test_loss=True,
    get_full_test_accuracy=True,
    train_loss_name="Base Train Loss",
    test_loss_name="Base Test Loss",
    test_accuracy_name="Base Test Accuracy",
)

hyper_scaled_train_config = BaseTrainConfig(
    num_epochs=1,
    train_loss_name="Hyper Scaled Train Loss",
)

hyper_binary_train_config = BaseTrainConfig(
    lr=0.0001,
    num_epochs=1,
    train_loss_name="Hyper Binary Train Loss",
    )


base_experiment_config = ExperimentConfig(
    project_name="Hypertraining",
    experiment="hypernet",
    model_type="base",
    model_dims=[784, 128, 10],
)

hyper_scaled_experiment_config = ExperimentConfig(
    project_name="Hypertraining",
    experiment="hypernet",
    model_type="hyper_scaled",
    model_dims=[3, 1024, 1],
)

hyper_binary_experiment_config = ExperimentConfig(
    project_name="Hypertraining",
    experiment="hypernet",
    model_type="hyper_binary",
    model_dims=[3, 64, 512, 64, 1],
    lr=0.0001,
)

train_loader, test_loader = get_dataloaders(
    base_experiment_config.dataset_name,
    base_train_config.batch_size,
    train_size=100,
    test_size=100,
)


base_model_estimate_scaled_path = "trained_models/base_mlp_estimate_scaled.t"
base_model_estimate_binary_path = "trained_models/base_mlp_estimate_binary.t"


if base_train_config.log_with_wandb:
    wandb.init(project="hypertraining")


base_model = BaseMLP(
    dimensions=base_experiment_config.model_dims,
    activation=base_experiment_config.model_act,
    )
hyper_model_scaled = base_model.get_hyper_model_scaled_input(hyper_scaled_experiment_config)
print(f"hyper config scaled dims: {hyper_scaled_experiment_config.model_dims}")
hyper_model_binary = base_model.get_hyper_model_binary_input(hyper_binary_experiment_config)
print(f"hyper config binary dims: {hyper_binary_experiment_config.model_dims}")


print(f"Number of parameters in base model: {base_model.num_parameters}")
print(
    f"Number of parameters in hyper model scaled: {hyper_model_scaled.num_parameters}"
)
print(
    f"Number of parameters in hyper model binary: {hyper_model_binary.num_parameters}"
)


try:
    base_model.load(base_experiment_config.model_path)
    print(f"File {base_experiment_config.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["base"]:
        print(f"File {base_experiment_config.model_path} not found. Training model...")
        base_train_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        base_test_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        base_model.train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            train_loss_fn=base_train_loss_fn,
            test_loss_fn=base_test_loss_fn,
            train_config=base_train_config,
        )
        base_model.save(base_experiment_config.model_dir, base_experiment_config.model_name)

try:
    hyper_model_scaled.load(hyper_scaled_experiment_config.model_path)
    print(f"File {hyper_scaled_experiment_config.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["hyper_scaled"]:
        print(
            f"File {hyper_scaled_experiment_config.model_path} not found. Training model..."
        )
        hyper_train_loss_fn = torch.nn.MSELoss(reduction="mean")
        hyper_test_loss_fn = torch.nn.MSELoss(reduction="sum")
        # hyper_scheduler = torch.optim.lr_scheduler.StepLR(hyper_optimizer, step_size=10, gamma=0.1)
        param_dataset_scaled = base_model.get_parameter_dataset(
            transform=base_model.scale_indices_transform
        )
        param_dataloader_scaled = torch.utils.data.DataLoader(
            dataset=param_dataset_scaled,
            batch_size=hyper_scaled_train_config.batch_size,
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

        hyper_model_scaled.train_model(
            train_loader=param_dataloader_scaled,
            test_loader=param_dataloader_scaled,
            train_loss_fn=hyper_train_loss_fn,
            test_loss_fn=hyper_test_loss_fn,
            train_config=hyper_scaled_train_config,
            callback=callback,
        )
        hyper_model_scaled.save(
            hyper_scaled_experiment_config.model_dir, hyper_scaled_experiment_config.model_name
        )
print("Hooray!")

try:
    hyper_model_binary.load(hyper_binary_experiment_config.model_path)
    print(f"File {hyper_binary_experiment_config.model_path} found. Loading model...")
except FileNotFoundError:
    if to_train["hyper_binary"]:
        print(
            f"File {hyper_binary_experiment_config.model_path} not found. Training model..."
        )
        hyper_train_loss_fn = torch.nn.MSELoss(reduction="mean")
        hyper_test_loss_fn = torch.nn.MSELoss(reduction="sum")
        param_dataset_binary = base_model.get_parameter_dataset(
            transform=base_model.binary_indices_transform
        )
        param_dataloader_binary = torch.utils.data.DataLoader(
            dataset=param_dataset_binary,
            batch_size=hyper_binary_train_config.batch_size,
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

        hyper_model_binary.train_model(
            train_loader=param_dataloader_binary,
            test_loader=param_dataloader_binary,
            train_loss_fn=hyper_train_loss_fn,
            test_loss_fn=hyper_test_loss_fn,
            train_config=hyper_binary_train_config,
            callback=callback,
        )
        hyper_model_binary.save(
            hyper_binary_experiment_config.model_dir, hyper_binary_experiment_config.model_name
        )
        reconstructed_accuracy = get_reconstructed_accuracy(
            base_model,
            hyper_model_binary,
            base_model.binary_indices_transform,
            test_loader,
        )
        print(f"Base model estimate scaled, accuracy: {reconstructed_accuracy}")
