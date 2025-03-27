from dataclasses import dataclass
from typing import Optional, List
import wandb
from math import prod
from torch.utils.data import DataLoader


@dataclass
class BaseTrainConfig:
    optimizer_name: str = "adam"
    lr: float = 0.01
    batch_size: int = 64
    dropout_prob: float = 0.0
    weight_decay: float = 0.0
    num_epochs: int = 100
    use_whole_dataset: bool = False
    use_early_stopping: bool = False
    target_full_train_loss: Optional[float] = 0.01
    patience: Optional[int] = 50

    get_full_train_loss: bool = False
    get_full_test_loss: bool = False
    get_full_train_accuracy: bool = False
    get_full_test_accuracy: bool = False

    get_final_train_loss: bool = False
    get_final_test_loss: bool = False
    get_final_train_accuracy: bool = False
    get_final_test_accuracy: bool = False
    
    train_loss_name: str = "Train Loss"
    test_loss_name: str = "Test Loss"
    train_accuracy_name: str = "Train Accuracy"
    test_accuracy_name: str = "Test Accuracy"

    def __post_init__(self):
        if self.use_early_stopping:
            if self.target_full_train_loss is None:
                raise ValueError(
                    "Must provide target_full_train_loss when use_early_stopping is True"
                )
            if self.patience is None:
                raise ValueError(
                    "Must provide patience when use_early_stopping is True"
                )
            if self.get_full_train_loss is False:
                raise ValueError(
                    "Must set get_full_train_loss to True when use_early_stopping is True"
                )


@dataclass
class BaseResults:
    full_train_accuracy: Optional[float]
    full_test_accuracy: Optional[float]
    full_train_loss: float
    full_test_loss: Optional[float]
    reached_target: bool
    epochs_taken: int
    lost_patience: bool
    ran_out_of_epochs: bool

    def __post_init__(self):
        self.generalization_gap = self.full_train_accuracy - self.full_test_accuracy
    
    def log(self, prefix=""):
        metrics = {
            f"{prefix}Train Accuracy": self.full_train_accuracy,
            f"{prefix}Test Accuracy": self.full_test_accuracy,
            f"{prefix}Train Loss": self.full_train_loss,
            f"{prefix}Test Loss": self.full_test_loss,
            f"{prefix}Generalization Gap": self.generalization_gap,
            f"{prefix}Reached Target": self.reached_target,
            f"{prefix}Epochs Taken": self.epochs_taken,
            f"{prefix}Lost Patience": self.lost_patience,
            f"{prefix}Ran Out Of Epochs": self.ran_out_of_epochs
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        wandb.log(metrics)


@dataclass
class DistTrainConfig:
    lr: float = 0.003  # Was 0.01 but for some models this was too high
    batch_size: int = 128
    max_epochs: int = 100000
    use_whole_dataset: bool = False

    dim_skip: int = 10
    min_hidden_dim: int = 1
    max_hidden_dim: int = 2000
    guess_hidden_dim: int = 128
    dist_activation: str = "relu"
    shift_logits: bool = False

    objective: str = "kl"
    reduction: str = "mean"
    k: Optional[int] = 10
    alpha: Optional[float] = 10**2
    use_scheduler: bool = False
    use_early_stopping: bool = False
    target_kl_on_train: Optional[float] = 0.01
    patience: Optional[int] = 100
    print_every: int = 1000

    get_full_kl_on_train_data: bool = True
    get_full_kl_on_test_data: bool = False
    get_full_accuracy_on_test_data: bool = False
    get_full_l2_on_test_data: bool = False

    get_final_kl_on_train_data: bool = True
    get_final_kl_on_test_data: bool = False
    get_final_accuracy_on_train_data: bool = False
    get_final_accuracy_on_test_data: bool = False
    get_final_l2_on_test_data: bool = False

    def __post_init__(self):
        valid_objectives = {"kl", "l2"}
        if self.objective not in valid_objectives:
            raise ValueError(
                f"Invalid objective: {self.objective}. Must be one of {valid_objectives}"
            )

        valid_reductions = {"mean", "sum"}
        if self.reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction: {self.reduction}. Must be one of {valid_reductions}"
            )

        if self.target_kl_on_train is not None:
            if self.objective != "kl":
                raise ValueError(
                    "target_kl_on_train is only valid when objective is 'kl'"
                )
            if self.get_full_kl_on_train_data is False:
                raise ValueError(
                    "Must set get_kl_on_train_data to True when target_kl_on_train is not None"
                )


@dataclass(frozen=True)
class DistDataLoaders:
    domain_train_loader: Optional[DataLoader]
    domain_test_loader: Optional[DataLoader]
    logit_train_loader: Optional[DataLoader]
    logit_test_loader: Optional[DataLoader]


@dataclass
class DistTrialResults:
    kl_on_train_data: Optional[float]
    reached_target: bool
    epochs_taken: int
    lost_patience: bool
    ran_out_of_epochs: bool

    def log(self, prefix=""):
        metrics = {
            f"{prefix}KL on Train Data": self.kl_on_train_data,
            f"{prefix}Reached Target": self.reached_target,
            f"{prefix}Epochs Taken": self.epochs_taken,
            f"{prefix}Lost Patience": self.lost_patience,
            f"{prefix}Ran Out Of Epochs": self.ran_out_of_epochs
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        wandb.log(metrics)


@dataclass
class DistFinalResults:
    complexity: int
    kl_on_train_data: Optional[float]
    kl_on_test_data: Optional[float]
    accuracy_on_train_data: Optional[float]
    accuracy_on_test_data: Optional[float]
    l2_on_test_data: Optional[float]

    def log(self, prefix=""):
        metrics = {
            f"{prefix}Complexity": self.complexity,
            f"{prefix}KL on Train Data": self.kl_on_train_data,
            f"{prefix}KL on Test Data": self.kl_on_test_data,
            f"{prefix}Accuracy on Train Data": self.accuracy_on_train_data,
            f"{prefix}Accuracy on Test Data": self.accuracy_on_test_data,
            f"{prefix}L2 on Test Data": self.l2_on_test_data
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        wandb.log(metrics)


@dataclass
class PACBResults:
    max_sigma: float
    noisy_error: float
    noise_trials: List[dict]
    total_num_sigmas: int
    pac_bound_inverse_kl: float
    pac_bound_pinsker: float

    def log(self):
        metrics = {
            "max_sigma": self.max_sigma,
            "noisy_error": self.noisy_error,
            "noise_trials": self.noise_trials,
            "total_num_sigmas": self.total_num_sigmas,
            "pac_bound_inverse_kl": self.pac_bound_inverse_kl,
            "pac_bound_pinsker": self.pac_bound_pinsker,
        }
        wandb.log(metrics)


# TODO: This really shouldn't include batchsize and lr, but the name depends on them. Maybe just pass the name?
@dataclass
class ExperimentConfig:
    experiment: str  # "low_rank", "hypernet", "distillation"
    model_type: str  # e.g. base, hyper_scaled, hyper_binary, low_rank, full, base, dist
    model_dims: List[int] = None
    optimizer_name: str = "adam"
    lr: float = 0.01
    batch_size: int = 64
    dropout_prob: float = 0.0
    weight_decay: float = 0.0

    dataset_name: str = "MNIST"
    new_data_shape: Optional[tuple[int, int]] = None
    model_act: str = "relu"
    model_name: Optional[str] = None

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

        self.model_root_dir = f"trained_models/{self.experiment}/{self.dataset_name}/{self.new_data_shape_str}"
        self.model_init_dir = f"{self.model_root_dir}/init"
        self.model_trained_dir = f"{self.model_root_dir}/{self.model_type}"
        self.metrics_dir = f"{self.model_root_dir}/metrics"
        if self.model_name is None:
            self.model_name = (
                f"{self.model_dims_str}_lr{self.lr}_bs{self.batch_size}_dp{self.dropout_prob}_wd{self.weight_decay}.t"
            )
        self.model_init_path = f"{self.model_init_dir}/{self.model_name}"
        self.model_trained_path = f"{self.model_trained_dir}/{self.model_name}"
        self.model_metrics_path = f"{self.metrics_dir}/{self.model_name}.csv"
