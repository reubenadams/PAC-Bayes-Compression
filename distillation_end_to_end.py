import os
import torch
import wandb
import argparse
import copy
import json
import pandas as pd
from dataclasses import asdict

from config import BaseHyperparamsConfig, BaseDataConfig, BaseStoppingConfig, BaseRecordsConfig, BaseConfig, DistHyperparamsConfig, DistSizeAndStoppingConfig, DistObjectiveConfig, DistRecordsConfig, DistDataConfig, DistConfig, ExperimentConfig, BaseResults, DistFinalResults, PACBConfig, PACBResults
from models import MLP
from load_data import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train and distill neural network models")
    parser.add_argument("--toy_run", action="store_true", help="Run with smaller dataset for testing")
    parser.add_argument("--device", type=str, default="cpu", help="Specify device (cuda/cpu)")
    parser.add_argument("--dataset", type=str, default="MNIST1D", help="Dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pickle", action="store_true", default=True, help="Pickle results locally (default: True)")
    return parser.parse_args()


def setup_environment(seed):
    torch.manual_seed(seed)
    os.environ["WANDB_SILENT"] = "true"


def get_base_config(quick_test: bool, dataset_name: str, device: str):
    hyperparams = BaseHyperparamsConfig.from_wandb_config(wandb.config)
    data_config = BaseDataConfig(dataset_name=dataset_name, device=device)
    data_config.add_sample_sizes(quick_test)
    data_config.add_dataloaders(hyperparams.batch_size)
    size_and_stopping = BaseStoppingConfig.create(quick_test)
    records = BaseRecordsConfig()
    return BaseConfig(
        hyperparams=hyperparams,
        data=data_config,
        size_and_stopping=size_and_stopping,
        records=records,
    )


def get_dist_config(quick_test: bool, dataset_name: str, use_whole_dataset: bool, device: str, seed: int, base_model: MLP):
    hyperparams = DistHyperparamsConfig()
    data_config = DistDataConfig(
        dataset_name=dataset_name,
        use_whole_dataset=True,
        device=device
    )
    data_config.add_sample_sizes(quick_test)
    data_config.add_dataloaders(
        batch_size=hyperparams.batch_size,
        seed=seed,
        base_model=base_model,
    )
    size_and_stopping = DistSizeAndStoppingConfig.create(quick_test)
    objective = DistObjectiveConfig()
    records = DistRecordsConfig()
    return DistConfig(
        hyperparams=hyperparams,
        size_and_stopping=size_and_stopping,
        objective=objective,
        records=records,
        data_loaders=data_config
    )


def get_pacb_config(quick_test: bool):
    return PACBConfig.create(quick_test)


# def get_base_run_params(toy_run):

#     base_run_params = {
#         "use_early_stopping": True,
#         "get_full_train_loss": True,
#         "get_final_train_loss": True,
#         "get_final_test_loss": True,
#         "get_final_train_accuracy": True,
#         "get_final_test_accuracy": True,
#         "train_loss_name": "Base Train Loss",
#         "test_loss_name": "Base Test Loss",
#         "train_accuracy_name": "Base Train Accuracy",
#         "test_accuracy_name": "Base Test Accuracy",
#         }

#     if toy_run:
#         base_run_params |= {
#             "train_size": 100,
#             "test_size": 100,
#             "num_epochs": 200,
#             "patience": 10,
#             "target_full_train_loss": 0.1
#         }
#     else:
#         base_run_params |= {
#             "train_size": None,
#             "test_size": None,
#             "num_epochs": 1000000,
#             "patience": 1000,
#             "target_full_train_loss": 0.01
#         }

#     return base_run_params


# def get_dist_run_params(toy_run):
#     if toy_run:
#         dist_run_params = {
#             "train_size": 100,
#             "test_size": 100,
#             "max_epochs": 10000,
#             "patience": 10,
#             "target_kl_on_train": 0.1,
#             "num_dist_attempts": 1
#         }
#     else:
#         dist_run_params = {
#             "train_size": None,
#             "test_size": None,
#             "max_epochs": 100000,
#             "patience": 100,
#             "target_kl_on_train": 0.01,
#             "num_dist_attempts": 5
#         }
#     return dist_run_params


# def get_pacb_run_params(toy_run):
#     if toy_run:
#         dist_run_params = {
#             "train_size": 100,
#             "test_size": 100,
#             "num_mc_samples_max_sigma": 10**2,
#             "num_mc_samples_pac_bound": 10**2,
#             "delta": 0.05
#         }
#     else:
#         dist_run_params = {
#             "train_size": None,
#             "test_size": None,
#             "num_mc_samples_max_sigma": 10**5,
#             "num_mc_samples_pac_bound": 10**6,
#             "delta": 0.05
#         }
#     return dist_run_params


# def get_hyperparams():
#     hyperparams = {
#         "optimizer": wandb.config.optimizer_name,
#         "hidden_layer_width": wandb.config.hidden_layer_width,
#         "num_hidden_layers": wandb.config.num_hidden_layers,
#         "learning_rate": wandb.config.lr,
#         "batch_size": wandb.config.batch_size,
#         "dropout_prob": wandb.config.dropout_prob,
#         "weight_decay": wandb.config.weight_decay,
#     }
#     return hyperparams


# def get_run_name(hyperparams):
#     run_name = (f"op{hyperparams['optimizer']}_"
#                 f"hw{hyperparams['hidden_layer_width']}_"
#                 f"nl{hyperparams['num_hidden_layers']}_"
#                 f"lr{hyperparams['learning_rate']}_"
#                 f"bs{hyperparams['batch_size']}_"
#                 f"dp{hyperparams['dropout_prob']}_"
#                 f"wd{hyperparams['weight_decay']}")
#     return run_name


# def get_base_train_config(hyperparams, base_run_params):
#     base_train_config = BaseConfig(
#         optimizer_name=hyperparams["optimizer"],
#         lr=hyperparams["learning_rate"],
#         batch_size=hyperparams["batch_size"],
#         dropout_prob=hyperparams["dropout_prob"],
#         weight_decay=hyperparams["weight_decay"],

#         num_epochs=base_run_params["num_epochs"],
#         use_early_stopping=base_run_params["use_early_stopping"],
#         target_full_train_loss=base_run_params["target_full_train_loss"],
#         patience=base_run_params["patience"],

#         get_full_train_loss=base_run_params["get_full_train_loss"],
#         get_full_train_accuracy=base_run_params["get_train_accuracy"],
#         get_full_test_accuracy=base_run_params["get_test_accuracy"],
#         get_final_train_loss = True,
#         get_final_test_loss = True,
#         get_final_train_accuracy = True,
#         get_final_test_accuracy = True,
#         train_loss_name=base_run_params,
#         test_loss_name=base_run_params,
#         train_accuracy_name=base_run_params["train_accuracy_name"],
#         test_accuracy_name=base_run_params["test_accuracy_name"],
#         )
#     return base_train_config


# def get_dist_train_config(dist_run_params):
#     dist_train_config = DistConfig(
#         max_epochs=dist_run_params["max_epochs"],
#         use_whole_dataset=True,
#         use_early_stopping=True,
#         target_kl_on_train=dist_run_params["target_kl_on_train"],
#         patience=dist_run_params["patience"],
#     )
#     return dist_train_config


def get_base_experiment_config(hyperparams, model_dims, dataset_name, model_name):
    base_experiment_config = ExperimentConfig(
        experiment="distillation",
        model_type="base",
        model_dims=model_dims,
        optimizer_name=hyperparams["optimizer"],
        lr=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        dropout_prob=hyperparams["dropout_prob"],
        weight_decay=hyperparams["weight_decay"],
        dataset_name=dataset_name,
        model_name=model_name,
    )
    return base_experiment_config


def get_dist_experiment_config(hyperparams, model_dims, dataset_name, model_name):
    dist_experiment_config = ExperimentConfig(
        experiment="distillation",
        model_type="dist",
        model_dims=model_dims,
        lr=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        dropout_prob=hyperparams["dropout_prob"],
        weight_decay=hyperparams["weight_decay"],
        dataset_name=dataset_name,
        model_name=model_name,
    )
    return dist_experiment_config


def initialize_wandb_and_configs(dataset_name, base_run_params, dist_run_params):
    """Initialize wandb and create configuration objects"""
    run = wandb.init()
    hyperparams = BaseHyperparamsConfig.from_wandb_config(wandb.config)
    # hyperparams = get_hyperparams()
    run.name = hyperparams.run_name()
    # run.name = get_run_name(hyperparams)
    run.save()

    model_dims = [wandb.config.input_dim] + [wandb.config.hidden_layer_width] * wandb.config.num_hidden_layers + [wandb.config.output_dim]
    model_name = wandb.run.name

    base_train_config = get_base_train_config(hyperparams, base_run_params)
    dist_train_config = get_dist_train_config(dist_run_params)
    base_experiment_config = get_base_experiment_config(hyperparams, model_dims, dataset_name, model_name)
    dist_experiment_config = get_dist_experiment_config(hyperparams, model_dims, dataset_name, model_name)

    return base_train_config, dist_train_config, base_experiment_config, dist_experiment_config


def train_base_model(
        base_experiment_config: ExperimentConfig,
        base_train_config: BaseConfig,
        base_run_params,
        device,
        seed: int,
    ) -> tuple[MLP, MLP, BaseResults]:

    torch.manual_seed(seed)
    init_model = MLP(
        dimensions=base_experiment_config.model_dims,
        activation=base_experiment_config.model_act,
        dropout_prob=base_experiment_config.dropout_prob,
        device=device
    )
    init_model.save(base_experiment_config.model_init_dir, base_experiment_config.model_name)
    base_model = copy.deepcopy(init_model)

    torch.manual_seed(seed)
    train_loader, test_loader = get_dataloaders(
        base_experiment_config.dataset_name,
        base_experiment_config.batch_size,
        train_size=base_run_params["train_size"],
        test_size=base_run_params["test_size"],
    )

    base_metrics = base_model.train_model(
        train_config=base_train_config,
        train_loader=train_loader,
        test_loader=test_loader,
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        full_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
    )

    if base_metrics.reached_target:  # Only save if model reached target train loss
        print("Model reached target train loss")
        base_model.save(base_experiment_config.model_trained_dir, base_experiment_config.model_name)
    else:
        print("Model failed to reach target train loss")

    return init_model, base_model, base_metrics


def train_dist_model(
        base_model: MLP,
        base_experiment_config: ExperimentConfig,
        dist_experiment_config: ExperimentConfig,
        dist_train_config: DistConfig,
        dist_run_params,
        device,
        seed: int,
    ) -> tuple[MLP, DistFinalResults]:

    base_model.eval()

    # This repetition is necessary as it uses a different batch size and uses the whole dataset
    # torch.manual_seed(seed)
    # train_loader, test_loader = get_dataloaders(
    #     dataset_name=base_experiment_config.dataset_name,
    #     batch_size=dist_train_config.batch_size,
    #     train_size=dist_run_params["train_size"],
    #     test_size=dist_run_params["test_size"],
    #     use_whole_dataset=dist_train_config.use_whole_dataset,
    #     device=device,
    # )

    dist_model, dist_metrics = base_model.get_dist_complexity(
        dist_train_config,
        domain_train_loader=train_loader,
        domain_test_loader=test_loader,
        num_attempts=dist_run_params["num_dist_attempts"],
    )

    dist_model.save(dist_experiment_config.model_trained_dir, dist_experiment_config.model_name)

    return dist_model, dist_metrics


def get_pac_bound(
        init_model: MLP,
        base_model: MLP,
        base_experiment_config: ExperimentConfig,
        pacb_run_params,
        seed: int,
) -> PACBResults:

    init_model.eval()
    base_model.eval()

    torch.manual_seed(seed)
    train_loader, _ = get_dataloaders(
        base_experiment_config.dataset_name,
        base_experiment_config.batch_size,
        train_size=pacb_run_params["train_size"],
        test_size=pacb_run_params["test_size"],
    )

    max_sigma, noisy_error, sigmas_tried, errors, total_num_sigmas = base_model.get_max_sigma(
        dataset=train_loader.dataset,
        target_error_increase=pacb_run_params["target_error_increase"],
        num_mc_samples=pacb_run_params["num_mc_samples_max_sigma"],
        )
    noise_trials = [{"sigma": sigma, "noisy_error": error} for sigma, error in zip(sigmas_tried, errors)]
    pac_bound_inverse_kl = base_model.pac_bayes_error_bound_inverse_kl(
        prior=init_model,
        sigma=max_sigma,
        dataloader=train_loader,
        num_mc_samples=pacb_run_params["num_mc_samples_pac_bound"],
        delta=pacb_run_params["delta"],
        num_union_bounds=total_num_sigmas,
        )
    pac_bound_pinsker = base_model.pac_bayes_error_bound_pinsker(
        prior=init_model,
        sigma=max_sigma,
        dataloader=train_loader,
        num_mc_samples=pacb_run_params["num_mc_samples_pac_bound"],
        delta=pacb_run_params["delta"],
        num_union_bounds=total_num_sigmas
        )
    
    print(f"{max_sigma=}, {noisy_error=}, {pac_bound_inverse_kl=}, {pac_bound_pinsker=}")
    print(f"{total_num_sigmas=}")

    pacb_metrics = PACBResults(
        max_sigma=max_sigma,
        noisy_error=noisy_error,
        noise_trials=noise_trials,
        total_num_sigmas=total_num_sigmas,
        pac_bound_inverse_kl=pac_bound_inverse_kl,
        pac_bound_pinsker=pac_bound_pinsker,
    )

    return pacb_metrics


def log_and_save_metrics(
        base_metrics: BaseResults,
        dist_metrics: DistFinalResults, 
        pacb_metrics: PACBResults, 
        base_experiment_config: ExperimentConfig,
    ):
    base_metrics.log(prefix="Base ")
    dist_metrics.log(prefix="Dist ")
    pacb_metrics.log()
    metrics = asdict(base_metrics) | asdict(dist_metrics) | asdict(pacb_metrics)
    df = pd.DataFrame([metrics])
    df.to_csv(base_experiment_config.results_path, index=False)


def main():

    quick_test = True
    device = "cpu"
    dataset_name = "MNIST1D"
    seed = 0

    # args = parse_args()
    # setup_environment(args.seed)
    setup_environment(seed)
    # device = torch.device(args.device)
    os.makedirs("trained_models", exist_ok=True)

    base_config = get_base_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        device=device
    )
    print("Training base model...")
    init_model, base_model, base_metrics = train_base_model(
        base_train_config=base_train_config,
        base_experiment_config=base_experiment_config,
        base_run_params=base_run_params,
        device=device,
    )


    dist_config = get_dist_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        use_whole_dataset=True,
        device=device,
        seed=seed,
        base_model=base_config.records.model
    )

    if base_metrics.reached_target:
        print("Base model reached target train loss")
    else:
        print("Base model failed to reach target train loss")
        return

    print("Distilling model...")
    dist_model, dist_metrics = train_dist_model(
            base_model=base_model,
            base_experiment_config=base_experiment_config,
            dist_experiment_config=dist_experiment_config,
            dist_train_config=dist_train_config,
            dist_run_params=dist_run_params,
            device=device,
            seed=seed,
        )

    pacb_config = get_pacb_config(quick_test=quick_test)
    print("Calculating PAC-Bayes bound...")
    pacb_metrics = get_pac_bound(
            init_model=init_model,
            base_model=base_model,
            base_experiment_config=base_experiment_config,
            pacb_run_params=pacb_run_params,
            seed=seed,
    )
    log_and_save_metrics(
        base_metrics=base_metrics,
        dist_metrics=dist_metrics,
        pacb_metrics=pacb_metrics,
        base_experiment_config=base_experiment_config,
        seed=seed,
    )

    # base_run_params = get_base_run_params(args.toy_run)
    # dist_run_params = get_dist_run_params(args.toy_run)
    # pacb_run_params = get_pacb_run_params(args.toy_run)
    # base_run_params = get_base_run_params(toy_run)
    # dist_run_params = get_dist_run_params(toy_run)
    # pacb_run_params = get_pacb_run_params(toy_run)

    base_train_config, dist_train_config, base_experiment_config, dist_experiment_config = initialize_wandb_and_configs(
        # dataset_name=args.dataset,
        dataset_name=dataset,
        base_run_params=base_run_params,
        dist_run_params=dist_run_params,
    )


if __name__ == "__main__":
    main()
