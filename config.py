import os
from dataclasses import dataclass
from typing import Optional

from math import prod


@dataclass
class Config:
    experiment: str  # "low_rank", "hypernet", "distillation"
    model_type: str  # e.g. base, hyper, low_rank, full, dist
    model_dims: tuple[int, ...]

    dataset: str = "MNIST"
    new_data_shape: Optional[tuple[int, int]] = None
    model_act: str = "relu"
    
    train_epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.001

    def __post_init__(self):

        valid_experiments = {"low_rank", "hypernet", "distillation"}
        if self.experiment not in valid_experiments:
            raise ValueError(f"Invalid experiment: {self.experiment}. Must be one of {valid_experiments}")

        valid_datasets = {"MNIST", "CIFAR100"}
        if self.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {self.dataset}. Must be one of {valid_datasets}")

        valid_activations = {"relu", "sigmoid", "tanh"}
        if self.model_act not in valid_activations:
            raise ValueError(f"Invalid activation: {self.model_act}. Must be one of {valid_activations}")

        if self.new_data_shape is None:
            if self.dataset == "MNIST":
                self.new_data_shape = (28, 28)
            elif self.dataset == "CIFAR100":
                self.new_data_shape = (32, 32)

        if not (self.experiment == "hypernet" and self.model_type in {"hyper_scaled", "hyper_binary"}):  # Dimension check for hypernetworks will have to occur elsewhere.
            if prod(self.new_data_shape) != self.model_dims[0]:
                raise ValueError(f"Data shape {self.new_data_shape} does not match model input size {self.model_dims[0]}")

        self.new_data_shape_str = "x".join(map(str, self.new_data_shape))
        self.model_dims_str = "x".join(map(str, self.model_dims))

        self.model_dir = f"trained_models/{self.experiment}/{self.dataset}/{self.new_data_shape_str}"
        self.model_name = f"{self.model_type}_{self.model_dims_str}_{self.model_act}.t"
        self.model_path = f"{self.model_dir}/{self.model_name}"

    def ensure_dir_exists(self):
        os.makedirs(self.model_dir, exist_ok=True)


base_mnist_config = Config(
    experiment="hypernet",
    model_type="base",
    model_dims=(784, 128, 10),
)

hyper_mnist_config_scaled = Config(
    experiment="hypernet",
    model_type="hyper_scaled",
    model_dims=(3, 1024, 1),
    learning_rate=0.01,
)

hyper_mnist_config_binary = Config(
    experiment="hypernet",
    model_type="hyper_binary",
    model_dims=(3, 64, 512, 64, 1),
    learning_rate=0.0001,
)

low_rank_mnist_config = Config(
    experiment="low_rank",
    model_type="low_rank",
    model_dims=(784, 128, 10),
)

low_rank_CIFAR100_config = Config(
    experiment="low_rank",
    model_type="low_rank",
    model_dims=(32**2, 100, 100),
    dataset="CIFAR100",
)

full_mnist_config = Config(
    experiment="distillation",
    model_type="full",
    model_dims=(4, 256, 10),
    new_data_shape=(2, 2),
)

dist_mnist_config = Config(
    experiment="distillation",
    model_type="dist",
    model_dims=(4, 4096, 10),
    new_data_shape=(2, 2),
)
