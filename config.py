import os
from dataclasses import dataclass
from typing import Optional, List
import wandb
from math import prod
import torch
from torch.utils.data import DataLoader

from load_data import get_dataloaders


@dataclass
class BaseHyperparamsConfig:
    optimizer_name: str
    hidden_layer_width: int
    num_hidden_layers: int
    lr: float
    batch_size: int
    dropout_prob: float
    weight_decay: float
    activation: str = "relu"

    @property
    def run_name(self):
        return f"op{self.optimizer_name}_hw{self.hidden_layer_width}_nl{self.num_hidden_layers}_lr{self.lr}_bs{self.batch_size}_dp{self.dropout_prob}_wd{self.weight_decay}"

    @classmethod
    def from_wandb_config(cls, config):
        return cls(
            optimizer_name=config.optimizer_name,
            hidden_layer_width=config.hidden_layer_width,
            num_hidden_layers=config.num_hidden_layers,
            lr=config.lr,
            batch_size=config.batch_size,
            dropout_prob=config.dropout_prob,
            weight_decay=config.weight_decay
        )

    def to_dict(self):
        return {
            "Base Activation": self.activation,
            "Base Optimizer": self.optimizer_name,
            "Base Hidden Layer Width": self.hidden_layer_width,
            "Base Num Hidden Layers": self.num_hidden_layers,
            "Base Learning Rate": self.lr,
            "Base Batch Size": self.batch_size,
            "Base Dropout Prob": self.dropout_prob,
            "Base Weight Decay": self.weight_decay,
        }


@dataclass
class BaseDataConfig:
    dataset_name: str
    device: str
    new_input_shape: Optional[tuple[int, int]] = None
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    train_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

    def add_sample_sizes(self, quick_test):
        if quick_test:
            self.train_size = 100
            self.test_size = 100
        else:
            self.train_size = None
            self.test_size = None

    def __post_init__(self):
        if self.new_input_shape is None:
            if self.dataset_name == "MNIST":
                self._new_input_shape = (28, 28)
            elif self.dataset_name == "CIFAR10":
                self._new_input_shape = (32, 32)
            elif self.dataset_name == "MNIST1D":
                self._new_input_shape = (40,)

    def add_dataloaders(self, batch_size):
        self.train_loader, self.test_loader = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=batch_size,
            train_size=self.train_size,
            test_size=self.test_size,
            new_input_shape=self.new_input_shape,
            device=self.device
        )

    def to_dict(self):
        return {
            "Base Dataset": self.dataset_name,
            "Base New Input Shape": self._new_input_shape,
            "Base Train Size": self.train_size,
            "Base Test Size": self.test_size,
        }


@dataclass
class BaseStoppingConfig:
    max_epochs: int
    use_early_stopping: bool
    target_full_train_loss: float
    patience: int

    @classmethod
    def quick_test(cls):
        return cls(
            max_epochs=1,  # TODO: Return to 100
            use_early_stopping=True,
            target_full_train_loss=1.5,
            patience=1
        )
    
    @classmethod
    def full_scale(cls):
        return cls(
            max_epochs=1000000,
            use_early_stopping=True,
            target_full_train_loss=0.01,
            patience=1000
        )

    @classmethod
    def create(cls, quick_test: bool):
        if quick_test:
            return cls.quick_test()
        else:
            return cls.full_scale()

    def to_dict(self):
        return {
            "Base Max Epochs": self.max_epochs,
            "Base Use Early Stopping": self.use_early_stopping,
            "Base Target Full Train Loss": self.target_full_train_loss,
            "Base Patience": self.patience,
        }


@dataclass
class BaseRecordsConfig:
    get_full_train_loss: bool = True
    get_full_test_loss: bool = False
    get_full_train_accuracy: bool = False
    get_full_test_accuracy: bool = False

    get_final_train_loss: bool = True
    get_final_test_loss: bool = True
    get_final_train_accuracy: bool = True
    get_final_test_accuracy: bool = True

    train_loss_name: str = "Base Train Loss"
    test_loss_name: str = "Base Test Loss"
    train_accuracy_name: str = "Base Train Accuracy"
    test_accuracy_name: str = "Base Test Accuracy"


@dataclass
class BaseConfig:
    hyperparams: BaseHyperparamsConfig
    data: BaseDataConfig
    stopping: BaseStoppingConfig
    records: BaseRecordsConfig

    @property
    def run_name(self):
        return self.hyperparams.run_name

    def __post_init__(self):
        if self.stopping.use_early_stopping:
            if self.stopping.target_full_train_loss is None:
                raise ValueError(
                    "Must provide target_full_train_loss when use_early_stopping is True"
                )
            if self.stopping.patience is None:
                raise ValueError(
                    "Must provide patience when use_early_stopping is True"
                )
            if self.records.get_full_train_loss is False:
                raise ValueError(
                    "Must set get_full_train_loss to True when use_early_stopping is True"
                )

        self.model_dims = [prod(self.data._new_input_shape)] + [self.hyperparams.hidden_layer_width] * self.hyperparams.num_hidden_layers + [10]
        self.model_name = self.hyperparams.run_name

        self.new_input_shape_str = "x".join(map(str, self.data._new_input_shape))
        self.model_dims_str = "x".join(map(str, self.model_dims))

        self.model_root_dir = f"models/{self.data.dataset_name}/{self.new_input_shape_str}"
        self.model_init_dir = f"{self.model_root_dir}/init"
        self.model_base_dir = f"{self.model_root_dir}/base"
        self.model_dist_dir = f"{self.model_root_dir}/dist"
        self.metrics_dir = f"{self.model_root_dir}/metrics"

        os.makedirs(self.model_init_dir, exist_ok=True)
        os.makedirs(self.model_base_dir, exist_ok=True)
        os.makedirs(self.model_dist_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.metrics_path = f"{self.metrics_dir}/{self.hyperparams.run_name}.csv"
    
    def to_dict(self):
        return self.data.to_dict() | self.hyperparams.to_dict() | self.stopping.to_dict()


@dataclass
class BaseResults:
    final_train_loss: float
    reached_target: bool
    epochs_taken: int
    lost_patience: bool
    ran_out_of_epochs: bool
    final_train_accuracy: Optional[float] = None
    final_test_accuracy: Optional[float] = None
    final_test_loss: Optional[float] = None

    def __post_init__(self):
        self.generalization_gap = self.final_train_accuracy - self.final_test_accuracy
    
    def to_dict(self):
        return {
            "Base Final Train Loss": self.final_train_loss,
            "Base Final Test Loss": self.final_test_loss,
            "Base Final Train Accuracy": self.final_train_accuracy,
            "Base Final Test Accuracy": self.final_test_accuracy,
            "Base Generalization Gap": self.generalization_gap,
            "Base Reached Target": self.reached_target,
            "Base Epochs Taken": self.epochs_taken,
            "Base Lost Patience": self.lost_patience,
            "Base Ran Out Of Epochs": self.ran_out_of_epochs,
        }
    
    def log(self):
        wandb_metrics = {k: v for k, v in self.to_dict().items() if isinstance(v, (int, float, torch.Tensor))}
        wandb.log(wandb_metrics)


@dataclass
class DistHyperparamsConfig:
    activation: str = "relu"
    lr: float = 0.003  # Was 0.01 but for some models this was too high

    # dim_skip: int = 10  # TODO: This isn't actually used, so it shouldn't be here.
    # min_hidden_dim: int = 1
    max_hidden_dim: int = 2048
    initial_guess_hidden_dim: int = 128

    def __post_init__(self):
        if self.initial_guess_hidden_dim <= 1:
            raise ValueError(f"Initial guess hidden dim {self.initial_guess_hidden_dim} must be greater than 1")
        if self.initial_guess_hidden_dim > self.max_hidden_dim:
            raise ValueError(f"Initial guess hidden dim {self.initial_guess_hidden_dim} must be less than or equal to max hidden dim {self.max_hidden_dim}")

    def to_dict(self):
        return {
            "Dist Activation": self.activation,
            "Dist Learning Rate": self.lr,
            # "Dist Dim Skip": self.dim_skip,
            # "Dist Min Hidden Dim": self.min_hidden_dim,
            "Dist Max Hidden Dim": self.max_hidden_dim,
            "Dist Initial Guess Hidden Dim": self.initial_guess_hidden_dim,
        }


@dataclass
class DistDataConfig:
    dataset_name: str
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    batch_size: int = None
    use_whole_dataset: Optional[bool] = None
    domain_train_loader: Optional[DataLoader] = None
    domain_test_loader: Optional[DataLoader] = None
    logit_train_loader: Optional[DataLoader] = None
    logit_test_loader: Optional[DataLoader] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.use_whole_dataset and self.batch_size is not None:
            raise ValueError(f"use_whole_dataset is {self.use_whole_dataset} but batch_size is not None: {self.batch_size} ")

    def add_sample_sizes(self, quick_test):
        if quick_test:
            self.train_size = 100
            self.test_size = 100
        else:
            self.train_size = None
            self.test_size = None

    def add_dataloaders(self, new_input_shape, base_model):
        self.domain_train_loader, self.domain_test_loader = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            train_size=self.train_size,
            test_size=self.test_size,
            new_input_shape=new_input_shape,
            use_whole_dataset=self.use_whole_dataset,
            device=self.device
        )
        self.logit_train_loader, self.logit_test_loader = base_model.get_logits_dataloaders(
            domain_train_loader=self.domain_train_loader,
            domain_test_loader=self.domain_test_loader,
            batch_size=self.batch_size,
            use_whole_dataset=self.use_whole_dataset,
            device=self.device,
        )

    def to_dict(self):
        return {
            "Dist Dataset": self.dataset_name,
            "Dist Train Size": self.train_size,
            "Dist Test Size": self.test_size,
            "Dist Batch Size": self.batch_size,
            "Dist Use Whole Dataset": self.use_whole_dataset,
        }


@dataclass
class DistStoppingConfig:
    max_epochs: int
    use_early_stopping: bool
    num_attempts: int
    target_kl_on_train: Optional[float] = None
    patience: Optional[int] = None
    print_every: int = 1000

    @classmethod
    def quick_test(cls):
        return cls(
            max_epochs=10000,
            use_early_stopping=True,
            target_kl_on_train=0.1,
            patience=2,
            num_attempts=1,
        )
    
    @classmethod
    def full_scale(cls):
        return cls(
            max_epochs=100000,
            use_early_stopping=True,
            target_kl_on_train=0.01,
            patience=100,
            num_attempts=5
        )

    @classmethod
    def create(cls, quick_test: bool):
        if quick_test:
            return cls.quick_test()
        else:
            return cls.full_scale()

    def to_dict(self):
        return {
            "Dist Max Epochs": self.max_epochs,
            "Dist Use Early Stopping": self.use_early_stopping,
            "Dist Target KL on Train": self.target_kl_on_train,
            "Dist Patience": self.patience,
            "Dist Num Attempts": self.num_attempts,
        }


@dataclass
class DistObjectiveConfig:
    objective_name: str = "kl"
    reduction: str = "mean"
    k: Optional[int] = None  # 10
    alpha: Optional[float] = None  # 10**2
    use_scheduler: bool = False
    shift_logits: bool = False

    def __post_init__(self):
        self.full_objective_name = f"{self.objective_name.upper()} {self.reduction}"

    def to_dict(self):
        return {
            "Dist Objective": self.full_objective_name,
            "Dist k": self.k,
            "Dist alpha": self.alpha,
            "Dist Use Scheduler": self.use_scheduler,
            "Dist Shift Logits": self.shift_logits,
        }


@dataclass
class DistRecordsConfig:
    get_full_kl_on_train_data: bool = True
    get_full_kl_on_test_data: bool = False
    get_full_accuracy_on_train_data: bool = False
    get_full_accuracy_on_test_data: bool = False
    # get_full_l2_on_test_data: bool = False

    get_final_kl_on_train_data: bool = True
    get_final_kl_on_test_data: bool = True
    get_final_accuracy_on_train_data: bool = True
    get_final_accuracy_on_test_data: bool = True
    # get_final_l2_on_test_data: bool = False

    train_kl_name: str = "Dist Train Mean KL"
    test_kl_name: str = "Dist Test Mean KL"
    train_accuracy_name: str = "Dist Train Accuracy"
    test_accuracy_name: str = "Dist Test Accuracy"



@dataclass
class DistConfig:
    hyperparams: DistHyperparamsConfig
    stopping: DistStoppingConfig
    objective: DistObjectiveConfig
    records: DistRecordsConfig
    data: DistDataConfig

    def __post_init__(self):
        valid_objectives = {"kl", "l2"}
        if self.objective.objective_name not in valid_objectives:
            raise ValueError(
                f"Invalid objective: {self.objective.objective_name}. Must be one of {valid_objectives}"
            )

        valid_reductions = {"mean", "sum"}
        if self.objective.reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction: {self.objective.reduction}. Must be one of {valid_reductions}"
            )

        if self.stopping.target_kl_on_train is not None:
            if self.objective.objective_name != "kl":
                raise ValueError(
                    "target_kl_on_train is only valid when objective is 'kl'"
                )
            if self.records.get_full_kl_on_train_data is False:
                raise ValueError(
                    "Must set get_kl_on_train_data to True when target_kl_on_train is not None"
                )

    def to_dict(self):
        return self.data.to_dict() | self.hyperparams.to_dict() | self.stopping.to_dict() | self.objective.to_dict()


@dataclass
class DistAttemptResults:
    reached_target: bool
    epochs_taken: int
    lost_patience: bool
    ran_out_of_epochs: bool
    mean_kl_on_train_data: Optional[float] = None

    def log(self):
        wandb_metrics = {"Dist Mean KL on Train Data": self.mean_kl_on_train_data}
        wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
        wandb.log(wandb_metrics)


@dataclass
class DistFinalResults:
    complexity: int
    mean_kl_on_train_data: Optional[float] = None
    mean_kl_on_test_data: Optional[float] = None
    accuracy_on_train_data: Optional[float] = None
    accuracy_on_test_data: Optional[float] = None
    l2_on_test_data: Optional[float] = None
    
    def to_dict(self):
        return {
            "Dist Complexity": self.complexity,
            "Dist Mean KL on Train Data": self.mean_kl_on_train_data,
            "Dist Mean KL on Test Data": self.mean_kl_on_test_data,
            "Dist Accuracy on Train Data": self.accuracy_on_train_data,
            "Dist Accuracy on Test Data": self.accuracy_on_test_data,
            "Dist L2 on Test Data": self.l2_on_test_data
        }

    def log(self):
        wandb_metrics = {k: v for k, v in self.to_dict().items() if isinstance(v, (int, float, torch.Tensor))}
        wandb.log(wandb_metrics)


@dataclass
class PACBConfig:
    num_mc_samples_max_sigma: int
    num_mc_samples_pac_bound: int
    delta: float = 0.05
    target_error_increase: float = 0.1

    @classmethod
    def quick_test(cls):
        return cls(
            num_mc_samples_max_sigma=10**2,
            num_mc_samples_pac_bound=10**2,
        )

    @classmethod
    def full_scale(cls):
        return cls(
            num_mc_samples_max_sigma=10**5,
            num_mc_samples_pac_bound=10**6,
        )
    
    @classmethod
    def create(cls, quick_test: bool):
        if quick_test:
            return cls.quick_test()
        else:
            return cls.full_scale()

    def to_dict(self):
        return {
            "PACB Num MC Samples for Max Sigma": self.num_mc_samples_max_sigma,
            "PACB Num MC Samples for PACB Bound": self.num_mc_samples_pac_bound,
            "PACB Delta": self.delta,
            "PACB Target Error Increase": self.target_error_increase,
        }


@dataclass
class PACBResults:
    sigma: float
    noisy_error: float
    noise_trials: list[dict]
    total_num_sigmas: int
    pac_bound_inverse_kl: float
    pac_bound_pinsker: float
    
    def to_dict(self):
        return {
            "PACB Sigma": self.sigma,
            "PACB Noisy Error": self.noisy_error,
            "PACB Noise Trials": self.noise_trials,
            "PACB Total Num Sigmas": self.total_num_sigmas,
            "PACB Bound Inverse kl": self.pac_bound_inverse_kl,
            "PACB Bound Pinsker": self.pac_bound_pinsker,
        }

    def log(self):
        wandb_metrics = {k: v for k, v in self.to_dict().items() if isinstance(v, (float, torch.Tensor))}
        wandb.log(wandb_metrics)


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
