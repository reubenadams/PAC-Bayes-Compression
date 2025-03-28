import os
from dataclasses import dataclass
from typing import Optional, List
import wandb
from math import prod
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

    # @classmethod
    # def quick_test(cls):
    #     return cls(
    #         train_size=100,
    #         test_size=100,
    #     )
    
    # @classmethod
    # def full_scale(cls):
    #     return cls(
    #         train_size=None,
    #         test_size=None,
    #     )
    
    # @classmethod
    # def create(cls, quick_test: bool):
    #     if quick_test:
    #         return cls.quick_test()
    #     else:
    #         return cls.full_scale()

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


@dataclass
class BaseStoppingConfig:
    num_epochs: int
    use_early_stopping: bool
    target_full_train_loss: float
    patience: int

    @classmethod
    def quick_test(cls):
        return cls(
            num_epochs=100,
            use_early_stopping=True,
            target_full_train_loss=1.5,
            patience=1
        )
    
    @classmethod
    def full_scale(cls):
        return cls(
            num_epochs=1000000,
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

        self.new_input_shape_str = "x".join(map(str, self.data._new_input_shape))
        self.model_dims = [self.data._new_input_shape[0]] + [self.hyperparams.hidden_layer_width] * self.hyperparams.num_hidden_layers + [10]
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

        self.model_name = self.hyperparams.run_name
        self.metrics_path = f"{self.metrics_dir}/{self.hyperparams.run_name}.csv"
        # self.model_init_path = f"{self.model_init_dir}/{self.model_name}"
        # self.model_base_path = f"{self.model_base_dir}/{self.model_name}"
        # self.model_dist_path = f"{self.model_dist_dir}/{self.model_name}"
        # self.model_metrics_path = f"{self.metrics_dir}/{self.model_name}.csv"


@dataclass
class BaseResults:
    full_train_loss: float
    reached_target: bool
    epochs_taken: int
    lost_patience: bool
    ran_out_of_epochs: bool
    full_train_accuracy: Optional[float] = None
    full_test_accuracy: Optional[float] = None
    full_test_loss: Optional[float] = None

    def __post_init__(self):
        self.generalization_gap = self.full_train_accuracy - self.full_test_accuracy
    
    def log(self, prefix=""):
        metrics = {
            f"{prefix}Train Accuracy": self.full_train_accuracy,
            f"{prefix}Test Accuracy": self.full_test_accuracy,
            f"{prefix}Train Loss": self.full_train_loss,
            f"{prefix}Test Loss": self.full_test_loss,
            f"{prefix}Generalization Gap": self.generalization_gap,
            # f"{prefix}Reached Target": self.reached_target,
            # f"{prefix}Epochs Taken": self.epochs_taken,
            # f"{prefix}Lost Patience": self.lost_patience,
            # f"{prefix}Ran Out Of Epochs": self.ran_out_of_epochs
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        wandb.log(metrics)


@dataclass
class DistHyperparamsConfig:
    lr: float = 0.003  # Was 0.01 but for some models this was too high
    batch_size: int = 128
    activation: str = "relu"

    dim_skip: int = 10
    min_hidden_dim: int = 1
    max_hidden_dim: int = 2000
    initial_guess_hidden_dim: int = 128


@dataclass
class DistDataConfig:
    dataset_name: str
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    use_whole_dataset: Optional[bool] = None
    domain_train_loader: Optional[DataLoader] = None
    domain_test_loader: Optional[DataLoader] = None
    logit_train_loader: Optional[DataLoader] = None
    logit_test_loader: Optional[DataLoader] = None
    device: Optional[str] = None

    def add_sample_sizes(self, quick_test):
        if quick_test:
            self.train_size = 100
            self.test_size = 100
        else:
            self.train_size = None
            self.test_size = None

    # @classmethod
    # def quick_test(cls):
    #     return cls(
    #         train_size=100,
    #         test_size=100,
    #     )
    
    # @classmethod
    # def full_scale(cls):
    #     return cls(
    #         train_size=None,
    #         test_size=None,
    #     )
    
    # @classmethod
    # def create(cls, quick_test: bool):
    #     if quick_test:
    #         return cls.quick_test()
    #     else:
    #         return cls.full_scale()

    def add_dataloaders(self, batch_size, new_input_shape, base_model):
        self.domain_train_loader, self.domain_test_loader = get_dataloaders(
            dataset_name=self.dataset_name,
            batch_size=batch_size,
            train_size=self.train_size,
            test_size=self.test_size,
            new_input_shape=new_input_shape,
            use_whole_dataset=self.use_whole_dataset,
            device=self.device
        )
        self.logit_train_loader, self.logit_test_loader = base_model.get_logits_dataloaders(
            domain_train_loader=self.domain_train_loader,
            domain_test_loader=self.domain_test_loader,
            batch_size=batch_size,
            use_whole_dataset=self.use_whole_dataset,
            device=self.device,
        )


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
            patience=10,
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


@dataclass
class DistObjectiveConfig:
    objective_name: str = "kl"
    reduction: str = "mean"
    k: Optional[int] = 10
    alpha: Optional[float] = 10**2
    use_scheduler: bool = False
    shift_logits: bool = False


@dataclass
class DistRecordsConfig:
    get_full_kl_on_train_data: bool = True
    get_full_kl_on_test_data: bool = False
    get_full_accuracy_on_train_data: bool = False
    get_full_accuracy_on_test_data: bool = False
    get_full_l2_on_test_data: bool = False

    get_final_kl_on_train_data: bool = True
    get_final_kl_on_test_data: bool = True
    get_final_accuracy_on_train_data: bool = True
    get_final_accuracy_on_test_data: bool = True
    get_final_l2_on_test_data: bool = False


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



# @dataclass
# class DistConfig:
    # lr: float = 0.003  # Was 0.01 but for some models this was too high
    # batch_size: int = 128
    # dist_activation: str = "relu"
    # max_epochs: int = 100000
    # use_whole_dataset: bool = False

    # dim_skip: int = 10
    # min_hidden_dim: int = 1
    # max_hidden_dim: int = 2000
    # guess_hidden_dim: int = 128
    # shift_logits: bool = False

    # objective: str = "kl"
    # reduction: str = "mean"
    # k: Optional[int] = 10
    # alpha: Optional[float] = 10**2
    # use_scheduler: bool = False
    # use_early_stopping: bool = False
    # target_kl_on_train: Optional[float] = 0.01
    # patience: Optional[int] = 100
    # print_every: int = 1000

    # get_full_kl_on_train_data: bool = True
    # get_full_kl_on_test_data: bool = False
    # get_full_accuracy_on_test_data: bool = False
    # get_full_l2_on_test_data: bool = False

    # get_final_kl_on_train_data: bool = True
    # get_final_kl_on_test_data: bool = False
    # get_final_accuracy_on_train_data: bool = False
    # get_final_accuracy_on_test_data: bool = False
    # get_final_l2_on_test_data: bool = False

    # def __post_init__(self):
    #     valid_objectives = {"kl", "l2"}
    #     if self.objective not in valid_objectives:
    #         raise ValueError(
    #             f"Invalid objective: {self.objective}. Must be one of {valid_objectives}"
    #         )

    #     valid_reductions = {"mean", "sum"}
    #     if self.reduction not in valid_reductions:
    #         raise ValueError(
    #             f"Invalid reduction: {self.reduction}. Must be one of {valid_reductions}"
    #         )

    #     if self.target_kl_on_train is not None:
    #         if self.objective != "kl":
    #             raise ValueError(
    #                 "target_kl_on_train is only valid when objective is 'kl'"
    #             )
    #         if self.get_full_kl_on_train_data is False:
    #             raise ValueError(
    #                 "Must set get_kl_on_train_data to True when target_kl_on_train is not None"
    #             )


@dataclass
class DistTrialResults:
    reached_target: bool
    epochs_taken: int
    lost_patience: bool
    ran_out_of_epochs: bool
    kl_on_train_data: Optional[float] = None

    def log(self, prefix=""):
        metrics = {
            f"{prefix}KL on Train Data": self.kl_on_train_data,
            # f"{prefix}Reached Target": self.reached_target,
            # f"{prefix}Epochs Taken": self.epochs_taken,
            # f"{prefix}Lost Patience": self.lost_patience,
            # f"{prefix}Ran Out Of Epochs": self.ran_out_of_epochs
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        wandb.log(metrics)


@dataclass
class DistFinalResults:
    complexity: int
    kl_on_train_data: Optional[float] = None
    kl_on_test_data: Optional[float] = None
    accuracy_on_train_data: Optional[float] = None
    accuracy_on_test_data: Optional[float] = None
    l2_on_test_data: Optional[float] = None

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


@dataclass
class PACBResults:
    max_sigma: float
    noisy_error: float
    noise_trials: list[dict]
    total_num_sigmas: int
    pac_bound_inverse_kl: float
    pac_bound_pinsker: float

    def log(self):
        metrics = {
            "max_sigma": self.max_sigma,
            "noisy_error": self.noisy_error,
            # "noise_trials": self.noise_trials,
            # "total_num_sigmas": self.total_num_sigmas,
            "pac_bound_inverse_kl": self.pac_bound_inverse_kl,
            "pac_bound_pinsker": self.pac_bound_pinsker,
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
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
