from __future__ import annotations
import inspect
import os
import json
from dataclasses import dataclass, asdict, field
from typing import Optional
import wandb
from math import prod
import torch
from torch.utils.data import DataLoader

from load_data import get_dataloaders, get_rand_domain_dataset_and_loader, get_max_l2_norm_data, get_logit_loader


@dataclass
class BaseHyperparamsConfig:
    optimizer_name: str
    hidden_layer_width: int
    num_hidden_layers: int
    lr: float
    batch_size: int
    dropout_prob: float
    weight_decay: float
    activation_name: str = "relu"

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
            "Base Activation": self.activation_name,
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
    data_filepath: Optional[str] = None
    _C_train_domain = None
    _C_train_data = None

    def __post_init__(self):
        if self.new_input_shape is None:
            if self.dataset_name == "MNIST":
                self._new_input_shape = (28, 28)
                self.input_range = 2  # Data range was [0, 1] but became [-1, 1] after normalization
            elif self.dataset_name == "CIFAR10":
                self._new_input_shape = (32, 32)
                self.input_range = 2  # Data range was [0, 1] but became [-1, 1] after normalization
            elif self.dataset_name == "MNIST1D":
                self._new_input_shape = (40,)
                self.input_range = 8  # Data range was [-inf, inf] but became [-4, 4] after clipping
        else:
            self._new_input_shape = self.new_input_shape

    def add_sample_sizes(self, quick_test):
        if quick_test:
            self.train_size = 100
            self.test_size = 100
        else:
            self.train_size = None
            self.test_size = None

    def add_dataloaders(self, batch_size):
        self.train_loader, self.test_loader, self.data_filepath = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=batch_size,
            train_size=self.train_size,
            test_size=self.test_size,
            new_input_shape=self.new_input_shape,
            use_whole_dataset=False,
            device=self.device
        )

    @property
    def C_train_domain(self):
        """Returns the maximum l2 norm of a point in [0, input_range]^input_dim."""
        if self._C_train_domain is None:
            input_dim = prod(self._new_input_shape)
            self._C_train_domain = self.input_range * torch.sqrt(torch.tensor(input_dim))
        return self._C_train_domain

    @property
    def C_train_data(self):
        if self._C_train_data is None:
            self._C_train_data = get_max_l2_norm_data(self.train_loader)
        return self._C_train_data

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
    target_full_train_loss: Optional[float]
    patience: Optional[int]

    def __post_init__(self):
        if self.use_early_stopping:
            if self.patience is None and self.target_full_train_loss is None:
                raise ValueError("Must provide one of patience or target_full_train_loss when use_early_stopping is True")

    @classmethod
    def quick_test(cls):
        return cls(
            max_epochs=100,
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
    experiment_type: str

    @property
    def run_name(self):
        return self.hyperparams.run_name

    def __post_init__(self):

        self.model_dims = [prod(self.data._new_input_shape)] + [self.hyperparams.hidden_layer_width] * self.hyperparams.num_hidden_layers + [10]
        self.model_name = self.hyperparams.run_name

        self.new_input_shape_str = "x".join(map(str, self.data._new_input_shape))
        self.model_dims_str = "x".join(map(str, self.model_dims))

        self.model_root_dir = f"{self.experiment_type}/models/{self.data.dataset_name}/{self.new_input_shape_str}"
        self.model_init_dir = f"{self.model_root_dir}/init"
        self.model_base_dir = f"{self.model_root_dir}/base"
        os.makedirs(self.model_init_dir, exist_ok=True)
        os.makedirs(self.model_base_dir, exist_ok=True)
        if self.experiment_type == "distillation":
            self.model_dist_dir = f"{self.model_root_dir}/dist"
            self.dist_metrics_dir = f"{self.model_root_dir}/dist_metrics"
            os.makedirs(self.model_dist_dir, exist_ok=True)
            os.makedirs(self.dist_metrics_dir, exist_ok=True)
            self.dist_metrics_path = f"{self.dist_metrics_dir}/{self.hyperparams.run_name}.csv"
        elif self.experiment_type == "quantization":
            self.quant_metrics_dir = f"{self.model_root_dir}/quant_metrics"  # This is here in addition to the below dirs because you're actually saving things twice.

            self.no_comp_metrics_dir = f"{self.model_root_dir}/no_comp_metrics"
            self.low_rank_only_metrics_dir = f"{self.model_root_dir}/low_rank_only_metrics"
            self.quant_only_metrics_dir = f"{self.model_root_dir}/quant_only_metrics"
            self.low_rank_and_quant_metrics_dir = f"{self.model_root_dir}/low_rank_and_quant_metrics"
            self.best_comp_metrics_dir = f"{self.model_root_dir}/best_comp_metrics"
            
            os.makedirs(self.quant_metrics_dir, exist_ok=True)
            
            os.makedirs(self.no_comp_metrics_dir, exist_ok=True)
            os.makedirs(self.low_rank_only_metrics_dir, exist_ok=True)
            os.makedirs(self.quant_only_metrics_dir, exist_ok=True)
            os.makedirs(self.low_rank_and_quant_metrics_dir, exist_ok=True)
            os.makedirs(self.best_comp_metrics_dir, exist_ok=True)

            self.quant_metrics_path = f"{self.quant_metrics_dir}/{self.hyperparams.run_name}.csv"  # This is here in addition to the below paths because you're actually saving things twice.
            
            self.no_comp_metrics_path = f"{self.no_comp_metrics_dir}/{self.hyperparams.run_name}.json"
            self.low_rank_only_metrics_path = f"{self.low_rank_only_metrics_dir}/{self.hyperparams.run_name}.json"
            self.quant_only_metrics_path = f"{self.quant_only_metrics_dir}/{self.hyperparams.run_name}.json"
            self.low_rank_and_quant_metrics_path = f"{self.low_rank_and_quant_metrics_dir}/{self.hyperparams.run_name}.json"
            self.best_comp_metrics_path = f"{self.best_comp_metrics_dir}/{self.hyperparams.run_name}.json"
        else:
            raise ValueError(f"Invalid experiment type: {self.experiment_type}. Must be 'distillation' or 'quantization'.")
    
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
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (int, float, torch.Tensor)}
        wandb.log(wandb_metrics)


@dataclass
class DistHyperparamsConfig:
    activation_name: str = "relu"
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
            "Dist Activation": self.activation_name,
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
    base_logit_train_loader: Optional[DataLoader] = None
    base_logit_test_loader: Optional[DataLoader] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.use_whole_dataset and self.batch_size is not None:
            raise ValueError(f"use_whole_dataset is {self.use_whole_dataset} but batch_size is not None: {self.batch_size} ")
        if not self.use_whole_dataset and self.batch_size is None:
            raise ValueError(f"use_whole_dataset is {self.use_whole_dataset} but batch_size is None")

    def add_sample_sizes(self, quick_test):
        if quick_test:
            self.train_size = 100
            self.test_size = 100
        else:
            self.train_size = None
            self.test_size = None

    # Add a dataset to distill on
    def add_dataloaders(self, new_input_shape, train_dataset=None, test_dataset=None, data_filepath=None):
        self.domain_train_loader, self.domain_test_loader, self.data_filepath = get_dataloaders(
            dataset_name=self.dataset_name,
            train_size=self.train_size,
            test_size=self.test_size,
            new_input_shape=new_input_shape,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            data_filepath=data_filepath,
            use_whole_dataset=self.use_whole_dataset,
            batch_size=self.batch_size,
            device=self.device
        )

    def add_base_logit_loaders(
            self,
            base_model,
            train_dataset,
            test_dataset,
            batch_size=None,
        ):
        self.base_logit_train_loader = get_logit_loader(
            model=base_model,
            dataset=train_dataset,
            batch_size=batch_size,
        )
        self.base_logit_test_loader = get_logit_loader(
            model=base_model,
            dataset=test_dataset,
            batch_size=batch_size,
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
        wandb_metrics = {k: v for k, v in wandb_metrics.items() if type(v) in (int, float, torch.Tensor)}
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
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (int, float, torch.Tensor)}
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
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (float, torch.Tensor)}
        wandb.log(wandb_metrics)


@dataclass
class CompConfig:
    
    # First three arguments to be passed from BaseConfig
    dataset_name: str
    device: str
    new_input_shape: Optional[tuple[int, int]]

    delta: float = 0.05
    min_rank: int = 1
    rank_step: int = 1
    max_codeword_length: int = 10
    get_low_rank_only_results: bool = True
    get_quant_only_results: bool = True
    get_low_rank_and_quant_results: bool = True
    compress_model_difference: bool = True

    use_whole_dataset: bool = True  # Note this will be used for all six dataloaders: train_loader, test_loader, rand_domain_loader, base_logit_train_loader, base_logit_test_loader, and base_logit_rand_domain_loader
    rand_domain_loader_batch_size: int = 128
    rand_domain_loader_sample_size: int = 10**6
    dist_min: Optional[float] = None
    dist_max: Optional[float] = None

    def __post_init__(self):
        if self.dataset_name == "MNIST1D":
            self.dist_min = -4.0
            self.dist_max = 4.0
        # TODO: This depends on the normalization done in load_data.py, so the normalization values should be passed to this config
        elif self.dataset_name in {"MNIST", "CIFAR10"}:
            self.dist_min = -1.0
            self.dist_max = 1.0
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}. Must be 'MNIST1D', 'MNIST', or 'CIFAR10'.")

        self.rand_domain_dataset, self.rand_domain_loader = get_rand_domain_dataset_and_loader(
            data_shape=self.new_input_shape,
            use_whole_dataset=self.use_whole_dataset,
            sample_size=self.rand_domain_loader_sample_size,
            batch_size=self.rand_domain_loader_batch_size,
            dist_min=self.dist_min,
            dist_max=self.dist_max,
            device=self.device,
        )

    @classmethod
    def create(
        cls,
        quick_test: bool,
        dataset_name: str,
        device: str,
        new_input_shape: Optional[tuple[int, int]]
    ) -> CompConfig:
        
        if quick_test:
            return cls(
                dataset_name=dataset_name,
                device=device,
                new_input_shape=new_input_shape,
                max_codeword_length=4,
                min_rank=3,
                rank_step=10**10,  # Large rank_step ensures only one low rank model is built
                rand_domain_loader_sample_size=10**3
            )
        else:
            return cls(
                dataset_name=dataset_name,
                device=device,
                new_input_shape=new_input_shape,
            )
    
    def to_dict(self):
        return {
            "Comp Delta": self.delta,
            "Comp Min Rank": self.min_rank,
            "Comp Rank Step": self.rank_step,
            "Comp Max Codeword Length": self.max_codeword_length,
            "Comp Get Low Rank Only Results": self.get_low_rank_only_results,
            "Comp Get Quant Only Results": self.get_quant_only_results,
            "Comp Get Low Rank and Quant Results": self.get_low_rank_and_quant_results,
            "Comp Compress Model Difference": self.compress_model_difference,
        }

    def add_dataloaders(self, train_dataset, test_dataset, data_filepath):
        self.train_loader, self.test_loader, self.data_filepath = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=None,
            train_size=None,
            test_size=None,
            new_input_shape=None,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            data_filepath=data_filepath,
            use_whole_dataset=self.use_whole_dataset,
            device=self.device
        )

    def add_base_logit_loaders(
            self,
            base_model,
            train_dataset,
            test_dataset,
            batch_size=None
        ):
        self.base_logit_train_loader = get_logits_loader(
            model=base_model,
            dataset=train_dataset,
            use_whole_dataset=self.use_whole_dataset,
            batch_size=batch_size,
        )
        self.base_logit_test_loader = get_logits_loader(
            model=base_model,
            dataset=test_dataset,
            use_whole_dataset=self.use_whole_dataset,
            batch_size=batch_size,
        )
        self.base_logit_rand_domain_loader = get_logits_loader(
            model=base_model,
            dataset=self.rand_domain_dataset,
            use_whole_dataset=self.use_whole_dataset,
            batch_size=batch_size,
        )


@dataclass
class CompResults:

    ranks: Optional[tuple[int]]
    codeword_length: Optional[int]
    
    C_domain: float
    C_data: float
    
    spectral_l2_bound_domain: float
    spectral_l2_bound_data: float
    empirical_l2_bound_domain: float
    empirical_l2_bound_train_data: float
    empirical_l2_bound_test_data: float
    
    margin_spectral_domain: float
    margin_spectral_data: float
    margin_empirical_domain: float
    margin_empirical_train_data: float
    margin_empirical_test_data: float
    
    train_accuracy: float
    test_accuracy: float
    
    train_margin_loss_spectral_domain: float
    train_margin_loss_spectral_data: float
    train_margin_loss_empirical_domain: float
    train_margin_loss_empirical_train_data: float
    train_margin_loss_empirical_test_data: float
    
    KL: float
    kl_bound: float
    
    error_bound_inverse_kl_spectral_domain: float
    error_bound_inverse_kl_spectral_data: float
    error_bound_inverse_kl_empirical_domain: float
    error_bound_inverse_kl_empirical_train_data: float
    error_bound_inverse_kl_empirical_test_data: float

    error_bound_pinsker_spectral_domain: float
    error_bound_pinsker_spectral_data: float
    error_bound_pinsker_empirical_domain: float
    error_bound_pinsker_empirical_train_data: float
    error_bound_pinsker_empirical_test_data: float

    def to_dict(self):
        """Returns a dictionary with formatted keys for wandb logging"""
        return {
            "Comp Ranks": self.ranks,
            "Comp Codeword Length": self.codeword_length,
            
            "Comp C Domain": self.C_domain,  # Stays fixed
            "Comp C Data": self.C_data,  # Stays fixed

            "Comp Spectral l2 Bound (using C_domain)": self.spectral_l2_bound_domain,
            "Comp Spectral l2 Bound (using C_data)": self.spectral_l2_bound_data,
            "Comp Empirical l2 Bound (on rand domain data)": self.empirical_l2_bound_domain,
            "Comp Empirical l2 Bound (on train data)": self.empirical_l2_bound_train_data,
            "Comp Empirical l2 Bound (on test data)": self.empirical_l2_bound_test_data,

            "Comp Margin Spectral (using C_domain)": self.margin_spectral_domain,
            "Comp Margin Spectral (using C_data)": self.margin_spectral_data,
            "Comp Empirical Margin (on rand domain data)": self.margin_empirical_domain,
            "Comp Empirical Margin (on train data)": self.margin_empirical_train_data,
            "Comp Empirical Margin (on test data)": self.margin_empirical_test_data,

            "Comp Train Accuracy": self.train_accuracy,
            "Comp Test Accuracy": self.test_accuracy,

            "Comp Train Margin Loss Spectral (using C_domain)": self.train_margin_loss_spectral_domain,
            "Comp Train Margin Loss Spectral (using C_data)": self.train_margin_loss_spectral_data,
            "Comp Train Margin Loss Empirical (on rand domain data)": self.train_margin_loss_empirical_domain,
            "Comp Train Margin Loss Empirical (on train data)": self.train_margin_loss_empirical_train_data,
            "Comp Train Margin Loss Empirical (on test data)": self.train_margin_loss_empirical_test_data,

            "Comp KL": self.KL,
            "Comp kl Bound": self.kl_bound,

            "Comp Error Bound Inverse kl Spectral (using C_domain)": self.error_bound_inverse_kl_spectral_domain,
            "Comp Error Bound Inverse kl Spectral (using C_data)": self.error_bound_inverse_kl_spectral_data,
            "Comp Error Bound Inverse kl Empirical (on rand domain data)": self.error_bound_inverse_kl_empirical_domain,
            "Comp Error Bound Inverse kl Empirical (on train data)": self.error_bound_inverse_kl_empirical_train_data,
            "Comp Error Bound Inverse kl Empirical (on test data)": self.error_bound_inverse_kl_empirical_test_data,
            
            "Comp Error Bound Pinsker Spectral (using C_domain)": self.error_bound_pinsker_spectral_domain,
            "Comp Error Bound Pinsker Spectral (using C_data)": self.error_bound_pinsker_spectral_data,
            "Comp Error Bound Pinsker Empirical (on rand domain data)": self.error_bound_pinsker_empirical_domain,
            "Comp Error Bound Pinsker Empirical (on train data)": self.error_bound_pinsker_empirical_train_data,
            "Comp Error Bound Pinsker Empirical (on test data)": self.error_bound_pinsker_empirical_test_data,
        }

    def log(self):
        """Log the metrics to wandb"""
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (int, float, torch.Tensor)}
        wandb.log(wandb_metrics)

    @classmethod 
    def get_extreme_vals(cls):
        """Returns an instance of CompResults with ranks and codeword_length set to None, train
        and test accuracy set to float('-inf'), and all other float fields set to float('inf')."""

        field_names = inspect.signature(cls).parameters
        field_values = dict()

        for field_name in field_names:
            if field_name in {'ranks', 'codeword_length'}:
                field_values[field_name] = None
            elif field_name in {'train_accuracy', 'test_accuracy'}:
                field_values[field_name] = float('-inf')
            else:
                field_values[field_name] = float('inf')
        
        return cls(**field_values)


@dataclass
class FinalCompResults:
    # Hierarchical structure: first by ranks, then by codeword_length
    results: dict[tuple[int], dict[int, CompResults]] = field(default_factory=dict)
    
    def get_best(self) -> None:
        """Returns a new CompResults object where each field is the best across all ranks and codeword lengths.
        For most fields, 'best' means lowest value. For train_accuracy and test_accuracy, 'best' means highest value."""
        if not self.results:
            raise ValueError("No results available to find best from")
        
        field_names = inspect.signature(CompResults).parameters
        best_values = asdict(CompResults.get_extreme_vals())

        for field_name in field_names:
            for ranks, rank_dict in self.results.items():
                for codeword_length, result in rank_dict.items():
                    value = getattr(result, field_name)
                    if field_name in {'ranks', 'codeword_length'}:
                        continue
                    if field_name in ['train_accuracy', 'test_accuracy']:
                        best_values[field_name] = max(best_values[field_name], value)
                    else:
                        best_values[field_name] = min(best_values[field_name], value)
        best_results = CompResults(**best_values)

        # Add the best results to the results dictionary with a special key
        self.results["best"] = dict()
        self.results["best"]["best"] = best_results
        self.best_results = best_results

    def to_dict(self):
        return self.best_results.to_dict()
    
    def log(self):
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (int, float, torch.Tensor)}
        wandb.log(wandb_metrics)

    def add_result(self, result: CompResults):
        # Initialize the inner dictionary if needed
        if result.ranks not in self.results:
            self.results[result.ranks] = dict()
        self.results[result.ranks][result.codeword_length] = result
    
    def save_to_json(self, filename: str):
        # Convert to dict with string keys for JSON compatibility
        data = {
                # Convert ranks tuple to comma-separated string
                self.str_ranks(ranks): {
                    str(codeword_length): asdict(rank_codeword_length_results)
                    for codeword_length, rank_codeword_length_results in rank_results.items()
                    }
                for ranks, rank_results in self.results.items()
            }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)
        
        experiment = cls()
        for ranks_str, rank_results_dict in data.items():
            ranks = tuple(int(r) for r in ranks_str.split(","))
            experiment.results[ranks] = dict()
            
            for codeword_length_str, rank_codeword_length_results_dict in rank_results_dict.items():
                codeword_length = int(codeword_length_str)
                if "ranks" in rank_codeword_length_results_dict and isinstance(rank_codeword_length_results_dict["ranks"], list):
                    rank_codeword_length_results_dict["ranks"] = tuple(rank_codeword_length_results_dict["ranks"])
                experiment.results[ranks][codeword_length] = CompResults(**rank_codeword_length_results_dict)
                
        return experiment
    
    def get_result(self, ranks: Optional[tuple[int]], codeword_length: int) -> CompResults:
        """Get results for a specific rank and codeword_length combination"""
        if ranks in self.results and codeword_length in self.results[ranks]:
            return self.results[ranks][codeword_length]
        else:
            raise ValueError(f"No results for {ranks=} and {codeword_length=}")

    @staticmethod
    def str_ranks(ranks):
        """Convert ranks tuple to a string (or None) for JSON compatibility"""
        if ranks is None or isinstance(ranks, str):
            return ranks
        if isinstance(ranks, tuple):
            return ",".join(map(str, ranks))
        raise ValueError(f"Invalid ranks type: {type(ranks)}. Expected tuple, None, or str.")


# TODO: This really shouldn't include batchsize and lr, but the name depends on them. Maybe just pass the name?
@dataclass
class ExperimentConfig:
    experiment: str  # "low_rank", "hypernet", "distillation"
    model_type: str  # e.g. base, hyper_scaled, hyper_binary, low_rank, full, base, dist
    model_dims: list[int] = None
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
