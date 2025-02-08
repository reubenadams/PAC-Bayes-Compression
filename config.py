from dataclasses import dataclass
from typing import Optional, List

from math import prod


@dataclass
class TrainConfig:
    lr: float = 0.01
    batch_size: int = 64
    num_epochs: int = 100
    use_early_stopping: bool = False
    target_overall_train_loss: Optional[float] = None
    patience: int = None

    log_with_wandb: bool = True
    get_overall_train_loss: bool = False
    get_test_loss: bool = False
    get_test_accuracy: bool = False
    train_loss_name: str = "Train Loss"
    test_loss_name: str = "Test Loss"
    test_accuracy_name: str = "Test Accuracy"

    def __post_init__(self):
        if self.use_early_stopping:
            if self.target_overall_train_loss is None:
                raise ValueError(
                    "Must provide target_overall_train_loss when use_early_stopping is True"
                )
            if self.patience is None:
                raise ValueError(
                    "Must provide patience when use_early_stopping is True"
                )
            if self.get_overall_train_loss is False:
                raise ValueError(
                    "Must set get_overall_train_loss to True when use_early_stopping is True"
                )


# TODO: This really shouldn't include batchsize and lr, but the name depends on them. Maybe just pass the name?
@dataclass
class ExperimentConfig:
    project_name: str
    experiment: str  # "low_rank", "hypernet", "distillation"
    model_type: str  # e.g. base, hyper_scaled, hyper_binary, low_rank, full, dist
    model_dims: List[int]
    lr: float = 0.01
    batch_size: int = 64

    dataset_name: str = "MNIST"
    new_data_shape: Optional[tuple[int, int]] = None
    model_act: str = "relu"

    def __post_init__(self):

        valid_experiments = {"low_rank", "hypernet", "distillation"}
        if self.experiment not in valid_experiments:
            raise ValueError(
                f"Invalid experiment: {self.experiment}. Must be one of {valid_experiments}"
            )

        valid_datasets = {"MNIST", "CIFAR10", "MNIST1D"}
        if self.dataset_name not in valid_datasets:
            raise ValueError(
                f"Invalid dataset: {self.dataset_name}. Must be one of {valid_datasets}"
            )

        valid_activations = {"relu", "sigmoid", "tanh"}
        if self.model_act not in valid_activations:
            raise ValueError(
                f"Invalid activation: {self.model_act}. Must be one of {valid_activations}"
            )

        if self.new_data_shape is not None and self.dataset_name == "MNIST1D":
            raise ValueError(f"MNIST1D does not support resizing.")

        if self.new_data_shape is None:
            if self.dataset_name == "MNIST":
                self.new_data_shape = (28, 28)
            elif self.dataset_name == "CIFAR10":
                self.new_data_shape = (32, 32)
            elif self.dataset_name == "MNIST1D":
                self.new_data_shape = (40,)

        if not (
            self.experiment == "hypernet"
            and self.model_type in {"hyper_scaled", "hyper_binary"}
        ):  # Dimension check for hypernetworks will have to occur elsewhere.
            data_input_dims = prod(self.new_data_shape)
            num_channels = 3 if self.dataset_name == "CIFAR10" else 1
            data_input_dims *= num_channels
            if data_input_dims != self.model_dims[0]:
                raise ValueError(
                    f"Data input dims {data_input_dims} does not match model input size {self.model_dims[0]}"
                )

        self.new_data_shape_str = "x".join(map(str, self.new_data_shape))
        self.model_dims_str = "x".join(map(str, self.model_dims))

        self.model_dir = (
            f"trained_models/{self.experiment}/{self.dataset_name}/{self.new_data_shape_str}"
        )
        self.model_name = f"{self.model_type}_{self.model_dims_str}_B{self.batch_size}_lr{self.lr}.t"
        self.model_path = f"{self.model_dir}/{self.model_name}"


# base_mnist_config = Config(
#     experiment="hypernet",
#     model_type="base",
#     model_dims=[784, 128, 10],
# )

# hyper_mnist_config_scaled = Config(
#     experiment="hypernet",
#     model_type="hyper_scaled",
#     model_dims=[3, 1024, 1],
#     lr=0.01,
# )

# hyper_mnist_config_binary = Config(
#     experiment="hypernet",
#     model_type="hyper_binary",
#     model_dims=[3, 64, 512, 64, 1],
#     lr=0.0001,
# )

# low_rank_mnist_config = Config(
#     experiment="low_rank",
#     model_type="low_rank",
#     model_dims=[784, 128, 10],
# )

# low_rank_CIFAR10_config = Config(
#     experiment="low_rank",
#     model_type="low_rank",
#     model_dims=[3 * 32**2, 100, 100],
#     dataset_name="CIFAR10",
# )

# full_mnist_config = Config(
#     experiment="distillation",
#     model_type="full",
#     model_dims=[4, 256, 10],
#     new_data_shape=(2, 2),
# )

# dist_kl_mnist_config = Config(
#     experiment="distillation",
#     model_type="dist",
#     model_dims=[4, 4096, 10],
#     new_data_shape=(2, 2),
#     epochs=100,
#     lr=0.1,
# )

# dist_l2_mnist_config = Config(
#     experiment="distillation",
#     model_type="dist",
#     model_dims=[4, 4096, 10],
#     new_data_shape=(2, 2),
#     epochs=100,
# )
