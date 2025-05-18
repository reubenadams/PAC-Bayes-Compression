from __future__ import annotations
import inspect
import os
import pandas as pd
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
    log_every: int = 10

    def __post_init__(self):
        if self.use_early_stopping:
            if self.patience is None and self.target_full_train_loss is None:
                raise ValueError("Must provide one of patience or target_full_train_loss when use_early_stopping is True")

    @classmethod
    def quick_test(cls):
        return cls(
            max_epochs=100,
            use_early_stopping=True,
            target_full_train_loss=2.0,
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
            self.comp_setup_dir = f"{self.model_root_dir}/quant_metrics"  # This is here in addition to the below dirs because you're actually saving things twice.

            self.no_comp_metrics_dir = f"{self.model_root_dir}/no_comp_metrics"
            self.quant_k_means_metrics_dir = f"{self.model_root_dir}/quant_k_means_metrics"
            self.quant_trunc_metrics_dir = f"{self.model_root_dir}/quant_trunc_metrics"
            
            self.low_rank_metrics_dir = f"{self.model_root_dir}/low_rank_metrics"
            self.low_rank_and_quant_k_means_metrics_dir = f"{self.model_root_dir}/low_rank_and_quant_k_means_metrics"
            self.low_rank_and_quant_trunc_metrics_dir = f"{self.model_root_dir}/low_rank_and_quant_trunc_metrics"
            
            self.best_comp_metrics_dir = f"{self.model_root_dir}/best_comp_metrics"
            
            os.makedirs(self.comp_setup_dir, exist_ok=True)
            
            os.makedirs(self.no_comp_metrics_dir, exist_ok=True)
            os.makedirs(self.quant_k_means_metrics_dir, exist_ok=True)
            os.makedirs(self.quant_trunc_metrics_dir, exist_ok=True)
            
            os.makedirs(self.low_rank_metrics_dir, exist_ok=True)
            os.makedirs(self.low_rank_and_quant_k_means_metrics_dir, exist_ok=True)
            os.makedirs(self.low_rank_and_quant_trunc_metrics_dir, exist_ok=True)
            
            os.makedirs(self.best_comp_metrics_dir, exist_ok=True)

            self.comp_setup_path = f"{self.comp_setup_dir}/{self.hyperparams.run_name}.csv"  # This is here in addition to the below paths because you're actually saving things twice.
            
            self.no_comp_metrics_path = f"{self.no_comp_metrics_dir}/{self.hyperparams.run_name}.json"
            self.quant_k_means_metrics_path = f"{self.quant_k_means_metrics_dir}/{self.hyperparams.run_name}.json"
            self.quant_trunc_metrics_path = f"{self.quant_trunc_metrics_dir}/{self.hyperparams.run_name}.json"
            
            self.low_rank_metrics_path = f"{self.low_rank_metrics_dir}/{self.hyperparams.run_name}.json"
            self.low_rank_and_quant_k_means_metrics_path = f"{self.low_rank_and_quant_k_means_metrics_dir}/{self.hyperparams.run_name}.json"
            self.low_rank_and_quant_trunc_metrics_path = f"{self.low_rank_and_quant_trunc_metrics_dir}/{self.hyperparams.run_name}.json"
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
            use_whole_dataset=self.use_whole_dataset,
            batch_size=batch_size,
        )
        self.base_logit_test_loader = get_logit_loader(
            model=base_model,
            dataset=test_dataset,
            use_whole_dataset=self.use_whole_dataset,
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
    log_every: int = 100

    @classmethod
    def quick_test(cls):
        return cls(
            max_epochs=10000,
            use_early_stopping=True,
            target_kl_on_train=0.1,
            patience=100,
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
    target_CE_loss_increase: float = 0.1
    delta: float = 0.05
    sigma_min: float = 2**(-14)
    sigma_max: float = 1
    num_mc_samples_sigma_target: int
    num_mc_samples_pac_bound: int

    def __post_init__(self):
        self.sigma_tol = self.sigma_min
        self.num_union_bounds = (self.sigma_max - self.sigma_min) / self.sigma_tol

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
            "PACB Num MC Samples Sigma Target": self.num_mc_samples_sigma_target,
            "PACB Num MC Samples for PACB Bound": self.num_mc_samples_pac_bound,
            "PACB Delta": self.delta,
            "PACB Target CE Loss Increase": self.target_CE_loss_increase,
        }


@dataclass
class PACBResults:
    sigma_target: float
    sigma_bound: float
    noisy_CE_loss: float
    # sigma: float
    # noisy_error: float
    # noise_trials: list[dict]
    # total_num_sigmas: int
    kl_bound: float
    error_bound_inverse_kl: float
    error_bound_pinsker: float
    
    def to_dict(self):
        return {
            "PACB Sigma Target": self.sigma_target,
            "PACB Sigma Bound": self.sigma_bound,
            "PACB Noisy CE Loss": self.noisy_CE_loss,
            # "PACB Sigma": self.sigma,
            # "PACB Noisy Error": self.noisy_error,
            # "PACB Noise Trials": self.noise_trials,
            # "PACB Total Num Sigmas": self.total_num_sigmas,
            "PACB KL Bound": self.kl_bound,
            "PACB Error Bound Inverse kl": self.error_bound_inverse_kl,
            "PACB Error Bound Pinsker": self.error_bound_pinsker,
        }

    def log(self):
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (float, torch.Tensor)}
        wandb.log(wandb_metrics)


@dataclass
class ComplexityMeasures:
    """Collection of complexity measures for a model."""
    # Confidence measures
    inverse_margin_tenth_percentile: float
    train_loss: float
    train_error: float
    output_entropy: float
    
    # Norm measures, all parameters
    l1_norm: float
    l2_norm: float
    l1_norm_from_init: float
    l2_norm_from_init: float

    # Norm measures, weights only
    spectral_sum: float
    spectral_product: float
    frobenius_sum: float
    frobenius_product: float
    spectral_sum_from_init: float
    spectral_product_from_init: float
    frobenius_sum_from_init: float
    frobenius_product_from_init: float

    # Sharpness measure
    inverse_squared_sigma_target: float
    
    # PAC-Bayes measures
    kl_bound_sigma_rounded: float
    error_bound_sigma_rounded_inverse_kl: float
    error_bound_sigma_rounded_pinsker: float

    # Distillation complexity
    min_hidden_width: int

    # Should be set to the same as PACBConfig.target_CE_loss_increase
    target_CE_loss_increase: float

    _use_log_x_axis = [
        "Inverse Margin Tenth Percentile",
        "L1 Norm",
        "L2 Norm",
        "L1 Norm From Init",
        "L2 Norm From Init",
        "Spectral Sum",
        "Spectral Product",
        "Frobenius Sum",
        "Frobenius Product",
        "Spectral Sum From Init",
        "Spectral Product From Init",
        "Frobenius Sum From Init",
        "Frobenius Product From Init",
        "KL Bound",
        "Error Bound Pinsker",
    ]

    def __post_init__(self):
        target_str = str(self.target_CE_loss_increase)
        # Dictionary mapping ordinary names to matplotlib names
        self._name_mapping = {
            "Inverse Margin Tenth Percentile": r"$\mu_{\text{inverse-margin}} = 1 / \gamma_{10\%}^2$",
            "Train Loss": r"$\mu_{\text{final-loss}} = \hat{L}_\text{cross-entropy}(h_{W, B})$",
            "Train Error": r"$\mu_{\text{final-error}} = \hat{L}_\text{0}(h_{W, B})$",
            "Output Entropy": r"$\mu_{\text{neg-entropy}} = \frac{1}{m}\sum_{i=1}^m H(h_{W, B}(x_i))$",
            
            "L1 Norm": r"$\mu_{\ell_1} = \|w\|_1$",
            "L2 Norm": r"$\mu_{\ell_2} = \|w\|_2$",
            "L1 Norm From Init": r"$\mu_{\ell_1\text{-init}} = \|w - w^0\|_1$",
            "L2 Norm From Init": r"$\mu_{\ell\text{-init}} = \|w - w^0\|_2$",
            
            "Spectral Sum": r"$\mu_{\text{spectral-sum}} = \sum_i \|W_i\|_{\text{spec}}$",
            "Spectral Product": r"$\mu_{\text{spectral-prod}} = \prod_i \|W_i\|_{\text{spec}}$",
            "Frobenius Sum": r"$\mu_{\text{frobenius-sum}} = \sum_i \|W_i\|_{\text{fro}}$",
            "Frobenius Product": r"$\mu_{\text{frobenius-prod}} = \prod_i \|W_i\|_{\text{fro}}$",
            
            "Spectral Sum From Init": r"$\mu_{\text{spectral-sum-init}} = \sum_i \|W_i - W_i^0\|_{\text{spec}}$",
            "Spectral Product From Init": r"$\mu_{\text{spectral-prod-init}} = \prod_i \|W_i - W_i^0\|_{\text{spec}}$",
            "Frobenius Sum From Init": r"$\mu_{\text{frobenius-sum-init}} = \sum_i \|W_i - W_i^0\|_{\text{fro}}$",
            "Frobenius Product From Init": r"$\mu_{\text{frobenius-prod-init}} = \prod_i \|W_i - W_i^0\|_{\text{fro}}$",
            
            "Inverse Squared Sigma Target": r"$\mu_{\text{sharpness}} = 1 / \sigma^2_{\beta=" + target_str + r"}$",  # TODO: I think the calculation is wrong as we're getting 0 or 1e12.
            
            "KL Bound": r"$\mu_{\text{pacb-kl-bound}} = \zeta(\tilde{\sigma}_{\beta=" + target_str + r"})$",  # TODO: You've put sigma_kl, but I think it's meant to be sigma_max?
            "Error Bound Inverse KL": r"$\mu_{\text{pacb-error-bound-inverse-kl}}$",
            "Error Bound Pinsker": r"$\mu_{\text{pacb-error-bound-pinsker}}$",
            
            "Dist Complexity": r"$\mu_{\text{dist-complexity}}$",
        }


    @classmethod
    def get_all_names(cls):
        return list(cls._name_mapping.keys())

    @classmethod
    def get_matplotlib_name(cls, name):
        if name not in cls._name_mapping:
            raise ValueError(f"Invalid name: {name}. Must be one of {cls._name_mapping.keys()}")
        return cls._name_mapping[name]

    @classmethod
    def use_log_x_axis(cls, name):
        if name in cls._use_log_x_axis:
            return True
        return False

    def to_dict(self):
        return {
            "Inverse Margin Tenth Percentile": self.inverse_margin_tenth_percentile,
            "Train Loss": self.train_loss,
            "Train Error": self.train_error,
            "Output Entropy": self.output_entropy,
            
            "L1 Norm": self.l1_norm,
            "L2 Norm": self.l2_norm,
            "L1 Norm From Init": self.l1_norm_from_init,
            "L2 Norm From Init": self.l2_norm_from_init,

            "Spectral Sum": self.spectral_sum,
            "Spectral Product": self.spectral_product,
            "Frobenius Sum": self.frobenius_sum,
            "Frobenius Product": self.frobenius_product,

            "Spectral Sum From Init": self.spectral_sum_from_init,
            "Spectral Product From Init": self.spectral_product_from_init,
            "Frobenius Sum From Init": self.frobenius_sum_from_init,
            "Frobenius Product From Init": self.frobenius_product_from_init,

            "Inverse Squared Sigma Target": self.inverse_squared_sigma_target,
            
            "KL Bound": self.kl_bound_sigma_rounded,
            "Error Bound Inverse KL": self.error_bound_sigma_rounded_inverse_kl,
            "Error Bound Pinsker": self.error_bound_sigma_rounded_pinsker,
            
            "Dist Complexity": self.min_hidden_width,
        }

    def log(self):
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (int, float, torch.Tensor)}
        wandb.log(wandb_metrics)


@dataclass
class EvaluationMetrics:
    complexity_measure_name: str
    rvalue: float
    r_squared: float
    pvalue: float
    krcc: float
    gkrcc_components: Optional[list[float]]
    gkrcc: float
    cit_k_zero_hyp_dims: float
    cit_k_one_hyp_dim: float
    cit_k_two_hyp_dims: float

    def __post_init__(self):
        self.cit_k_at_most_one_hyp_dim = min(self.cit_k_zero_hyp_dims, self.cit_k_one_hyp_dim)
        self.cit_k_at_most_two_hyp_dims = min(self.cit_k_zero_hyp_dims, self.cit_k_one_hyp_dim, self.cit_k_two_hyp_dims)

    def to_dict(self):
        return {
            "Complexity Measure Name": self.complexity_measure_name,
            "R Value": self.rvalue,
            "R Squared": self.r_squared,
            "P Value": self.pvalue,
            "KRCC": self.krcc,
            "GKRCC Components": self.gkrcc_components,
            "GKRCC": self.gkrcc,
            "CIT Zero Hyp Dims": self.cit_k_zero_hyp_dims,
            "CIT One Hyp Dim": self.cit_k_one_hyp_dim,
            "CIT Two Hyp Dims": self.cit_k_two_hyp_dims,
            "CIT At Most One Hyp Dim": self.cit_k_at_most_one_hyp_dim,
            "CIT At Most Two Hyp Dims": self.cit_k_at_most_two_hyp_dims,
        }


@dataclass
class CompConfig:
    
    # First three arguments to be passed from BaseConfig
    dataset_name: str
    device: str
    new_input_shape: Optional[tuple[int, int]]

    delta: float = 0.05
    min_rank: int = 1
    min_num_rank_values: int = 8
    max_codeword_length: int = 20
    max_codeword_length_for_low_rank: int = 10

    get_no_comp_results: bool = True
    get_quant_k_means_results: bool = True
    get_quant_trunc_results: bool = True
    
    get_low_rank_results: bool = True
    get_low_rank_and_quant_k_means_results: bool = True
    get_low_rank_and_quant_trunc_results: bool = True
    
    compress_model_difference: bool = True

    # Note this will be used for all six dataloaders: train_loader, test_loader, rand_domain_loader, base_logit_train_loader, base_logit_test_loader, and base_logit_rand_domain_loader
    use_whole_dataset: bool = None
      
    rand_domain_loader_batch_size: Optional[int] = None
    rand_domain_loader_sample_size: int = 10**1  # TODO: Set this in a more sensible way. Currently set to 10 because it's not actually used.
    dist_min: Optional[float] = None
    dist_max: Optional[float] = None

    def __post_init__(self):
        if self.dataset_name == "MNIST1D":
            self.dist_min = -4.0
            self.dist_max = 4.0
            if self.use_whole_dataset is None:
                self.use_whole_dataset = True
        # TODO: This depends on the normalization done in load_data.py, so the normalization values should be passed to this config
        elif self.dataset_name in {"MNIST", "CIFAR10"}:
            self.dist_min = -1.0
            self.dist_max = 1.0
            if self.use_whole_dataset is None:
                self.use_whole_dataset = False
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
            "Comp Max Codeword Length": self.max_codeword_length,

            "Comp Get No Comp Results": self.get_no_comp_results,
            "Comp Get Quant k-Means Results": self.get_quant_k_means_results,
            "Comp Get Quant Trunc Results": self.get_quant_trunc_results,
            "Comp Get Low Rank Results": self.get_low_rank_results,
            "Comp Get Low Rank and Quant k-Means Results": self.get_low_rank_and_quant_k_means_results,
            "Comp Get Low Rank and Quant Trunc Results": self.get_low_rank_and_quant_trunc_results,
            
            "Comp Compress Model Difference": self.compress_model_difference,
        }

    def add_dataloaders(self, train_dataset, test_dataset, data_filepath):
        self.train_loader, self.test_loader, self.data_filepath = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=128,
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
            batch_size=128,
        ):
        self.base_logit_train_loader = get_logit_loader(
            model=base_model,
            dataset=train_dataset,
            use_whole_dataset=self.use_whole_dataset,
            batch_size=batch_size,
        )
        self.base_logit_test_loader = get_logit_loader(
            model=base_model,
            dataset=test_dataset,
            use_whole_dataset=self.use_whole_dataset,
            batch_size=batch_size,
        )
        self.base_logit_rand_domain_loader = get_logit_loader(
            model=base_model,
            dataset=self.rand_domain_dataset,
            use_whole_dataset=self.use_whole_dataset,
            batch_size=batch_size,
        )


@dataclass
class CompResults:

    ranks: Optional[tuple[int]]
    codeword_length: Optional[int]
    exponent_bits: Optional[int]
    mantissa_bits: Optional[int]
    
    C_domain: float
    C_data: float
    
    spectral_l2_bound_domain: float
    margin_spectral_domain: float
    train_margin_loss_spectral_domain: float
    error_bound_inverse_kl_spectral_domain: float
    error_bound_pinsker_spectral_domain: float

    train_accuracy: float
    test_accuracy: float

    KL: float
    kl_bound: float

    spectral_l2_bound_data: float = None
    empirical_l2_bound_domain: float = None
    empirical_l2_bound_train_data: float = None
    empirical_l2_bound_test_data: float = None
    
    margin_spectral_data: float = None
    margin_empirical_domain: float = None
    margin_empirical_train_data: float = None
    margin_empirical_test_data: float = None
    
    
    train_margin_loss_spectral_data: float = None
    train_margin_loss_empirical_domain: float = None
    train_margin_loss_empirical_train_data: float = None
    train_margin_loss_empirical_test_data: float = None
    
    
    error_bound_inverse_kl_spectral_data: float = None
    error_bound_inverse_kl_empirical_domain: float = None
    error_bound_inverse_kl_empirical_train_data: float = None
    error_bound_inverse_kl_empirical_test_data: float = None

    error_bound_pinsker_spectral_data: float = None
    error_bound_pinsker_empirical_domain: float = None
    error_bound_pinsker_empirical_train_data: float = None
    error_bound_pinsker_empirical_test_data: float = None

    def __post_init__(self):
        check_comp_arguments(
            codeword_length=self.codeword_length,
            exponent_bits=self.exponent_bits,
            mantissa_bits=self.mantissa_bits,
        )

    def to_dict(self):
        """Returns a dictionary with formatted keys for wandb logging"""
        return {
            "Comp Ranks": self.ranks,
            "Comp Codeword Length": self.codeword_length,
            "Comp Exponent Bits": self.exponent_bits,
            "Comp Mantissa Bits": self.mantissa_bits,

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
    def get_extreme_initialization(cls):
        """Returns an instance of CompResults with ranks and codeword_length set to None, train
        and test accuracy set to float('-inf'), and all other float fields set to float('inf')."""

        field_names = inspect.signature(cls).parameters
        field_values = dict()

        for field_name in field_names:
            if field_name in {'ranks', 'codeword_length', "exponent_bits", 'mantissa_bits'}:
                field_values[field_name] = None
            elif field_name in {'train_accuracy', 'test_accuracy'}:
                field_values[field_name] = float('-inf')
            else:
                field_values[field_name] = float('inf')
        
        return cls(**field_values)


@dataclass
class FinalCompResults:
    """Stores multiple CompResults objects as a list, and provides methods to find the best results."""
    compression_scheme: str
    num_union_bounds: int
    all_results: list[CompResults] = field(default_factory=list)
    best_results: Optional[CompResults] = None

    def add_results(self, results: CompResults):
        self.all_results.append(results)

    def get_best_results(self) -> None:
        """Populates the best_results field with the best results for each compression scheme."""
        if len(self.all_results) == 0:
            raise ValueError("No results available to find best from")
        
        field_names = inspect.signature(CompResults).parameters
        best_values = asdict(CompResults.get_extreme_initialization())

        for field_name in field_names:
            all_vals_are_none = all([getattr(results, field_name) is None for results in self.all_results])
            if all_vals_are_none:
                best_values[field_name] = None
            else:
                for results in self.all_results:
                    value = getattr(results, field_name)
                    if field_name in {'ranks', 'codeword_length', 'exponent_bits', 'mantissa_bits'}:
                        best_values[field_name] = None
                    elif field_name in ['train_accuracy', 'test_accuracy']:
                        best_values[field_name] = max(best_values[field_name], value)
                    else:
                        if value is not None:
                            best_values[field_name] = min(best_values[field_name], value)
        self.best_results = CompResults(**best_values)  # This will have None for all compression parameters.

    @property
    def best_inverse_kl_results(self) -> tuple[CompResults, bool]:
        """Returns the results with the lowest inverse kl bound."""
        if len(self.all_results) == 0:
            raise ValueError("No results available to find best from")
        
        inverse_kl_bounds = [results.error_bound_inverse_kl_spectral_domain for results in self.all_results]
        all_equal = len(set(inverse_kl_bounds)) == 1
        index_of_best = inverse_kl_bounds.index(min(inverse_kl_bounds))
        return self.all_results[index_of_best], all_equal

    @property
    def best_pinsker_results(self) -> tuple[CompResults, bool]:
        """Returns the results with the lowest pinsker bound."""
        if len(self.all_results) == 0:
            raise ValueError("No results available to find best from")
        
        pinsker_bounds = [results.error_bound_pinsker_spectral_domain for results in self.all_results]
        all_equal = len(set(pinsker_bounds)) == 1
        index_of_best = pinsker_bounds.index(min(pinsker_bounds))
        return self.all_results[index_of_best], all_equal

    def to_dict(self):
        return self.best_results.to_dict()
    
    def log(self):
        wandb_metrics = {k: v for k, v in self.to_dict().items() if type(v) in (int, float, torch.Tensor)}
        wandb.log(wandb_metrics)

    def save_to_json(self, filepath: str):
        data = {
            "compression_scheme": self.compression_scheme,
            "num_union_bounds": self.num_union_bounds,
            "all_results": [asdict(result) for result in self.all_results],
            "best_results": asdict(self.best_results),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        experiment = cls(
            compression_scheme=data["compression_scheme"],
            num_union_bounds=data["num_union_bounds"],
            all_results=[CompResults(**result) for result in data["all_results"]],
            best_results=CompResults(**data["best_results"]),
        )
        return experiment


def check_comp_arguments(
        codeword_length: Optional[int],
        exponent_bits: Optional[int],
        mantissa_bits: Optional[int],
    ) -> None:
    if (exponent_bits is not None) and (mantissa_bits is not None):
        trunc = True
    elif (exponent_bits is None) and (mantissa_bits is None):
        trunc = False
    else:
        raise ValueError(f"Both {exponent_bits=} and {mantissa_bits=} must be None or both must be set.")
    if codeword_length is not None and trunc:
        raise ValueError(f"Cannot both quantize with {codeword_length=} and truncate with {exponent_bits=}, {mantissa_bits=}")