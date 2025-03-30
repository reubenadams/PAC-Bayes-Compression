import os
import torch
import wandb
import argparse
import copy
import pandas as pd
from dataclasses import asdict

import config
from models import MLP


def parse_args():
    parser = argparse.ArgumentParser(description="Train and distill neural network models")
    parser.add_argument("--toy_run", action="store_true", help="Run with smaller dataset for testing")
    parser.add_argument("--device", type=str, default="cpu", help="Specify device (cuda/cpu)")
    parser.add_argument("--dataset", type=str, default="MNIST1D", help="Dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pickle", action="store_true", default=True, help="Pickle results locally (default: True)")
    return parser.parse_args()


def get_base_config(quick_test: bool, dataset_name: str, device: str):
    hyperparams = config.BaseHyperparamsConfig.from_wandb_config(wandb.config)
    data_config = config.BaseDataConfig(dataset_name=dataset_name, device=device)
    data_config.add_sample_sizes(quick_test)
    data_config.add_dataloaders(hyperparams.batch_size)
    stopping_config = config.BaseStoppingConfig.create(quick_test)
    records = config.BaseRecordsConfig()
    return config.BaseConfig(
        hyperparams=hyperparams,
        data=data_config,
        stopping=stopping_config,
        records=records,
    )


def get_dist_config(
        quick_test: bool,
        dataset_name: str,
        use_whole_dataset: bool,
        device: str,
        base_config: config.BaseConfig,
        base_model: MLP
    ):
    hyperparams = config.DistHyperparamsConfig()
    data_config = config.DistDataConfig(
        dataset_name=dataset_name,
        use_whole_dataset=use_whole_dataset,
        device=device
    )
    data_config.add_sample_sizes(quick_test)
    data_config.add_dataloaders(
        new_input_shape=base_config.data.new_input_shape,
        base_model=base_model,
    )
    stopping_config = config.DistStoppingConfig.create(quick_test)
    objective = config.DistObjectiveConfig()
    records = config.DistRecordsConfig()
    return config.DistConfig(
        hyperparams=hyperparams,
        stopping=stopping_config,
        objective=objective,
        records=records,
        data=data_config
    )


def get_pacb_config(quick_test: bool):
    return config.PACBConfig.create(quick_test)


def train_base_model(
        base_config: config.BaseConfig,
    ) -> tuple[MLP, MLP, config.BaseResults]:

    init_model = MLP(
        dimensions=base_config.model_dims,
        activation=base_config.hyperparams.activation,
        dropout_prob=base_config.hyperparams.dropout_prob,
        device=base_config.data.device,
    )
    base_model = copy.deepcopy(init_model)
    print("init_model: ")
    print(init_model)
    print("base_model: ")
    print(base_model)

    base_metrics = base_model.train_model(
        base_config=base_config,
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        full_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
    )

    return init_model, base_model, base_metrics


def train_dist_model(
        base_model: MLP,
        dist_config: config.DistConfig,
    ) -> tuple[MLP, config.DistFinalResults]:

    base_model.eval()
    dist_model, dist_metrics = base_model.get_dist_complexity(dist_config=dist_config)
    return dist_model, dist_metrics


def get_pac_bound(
        init_model: MLP,
        base_model: MLP,
        base_config: config.BaseConfig,
        pacb_config: config.PACBConfig,
) -> config.PACBResults:

    init_model.eval()
    base_model.eval()

    max_sigma, noisy_error, sigmas_tried, errors, total_num_sigmas = base_model.get_max_sigma(
        dataset=base_config.data.train_loader.dataset,
        target_error_increase=pacb_config.target_error_increase,
        num_mc_samples=pacb_config.num_mc_samples_max_sigma,
        )
    noise_trials = [{"sigma": sigma, "noisy_error": error} for sigma, error in zip(sigmas_tried, errors)]
    pac_bound_inverse_kl = base_model.pac_bayes_error_bound_inverse_kl(
        prior=init_model,
        sigma=max_sigma,
        dataloader=base_config.data.train_loader,
        num_mc_samples=pacb_config.num_mc_samples_pac_bound,
        delta=pacb_config.delta,
        num_union_bounds=total_num_sigmas,
        )
    pac_bound_pinsker = base_model.pac_bayes_error_bound_pinsker(
        prior=init_model,
        sigma=max_sigma,
        dataloader=base_config.data.train_loader,
        num_mc_samples=pacb_config.num_mc_samples_pac_bound,
        delta=pacb_config.delta,
        num_union_bounds=total_num_sigmas
        )
    
    print(f"{max_sigma=}, {noisy_error=}, {pac_bound_inverse_kl=}, {pac_bound_pinsker.item()=}")
    print(f"{total_num_sigmas=}")

    pacb_metrics = config.PACBResults(
        sigma=max_sigma,
        noisy_error=noisy_error,
        noise_trials=noise_trials,
        total_num_sigmas=total_num_sigmas,
        pac_bound_inverse_kl=pac_bound_inverse_kl,
        pac_bound_pinsker=pac_bound_pinsker.item(),
    )

    return pacb_metrics


def log_and_save_metrics(
        run_id: str,
        base_config: config.BaseConfig,
        dist_config: config.DistConfig,
        pacb_config: config.PACBConfig,
        base_metrics: config.BaseResults,
        dist_metrics: config.DistFinalResults, 
        pacb_metrics: config.PACBResults,
    ):
    base_metrics.log()
    dist_metrics.log(prefix="Dist ")
    pacb_metrics.log()
    all_csv_values = {"Run ID": run_id, "Run Name": base_config.run_name} | base_config.to_dict() | base_metrics.to_dict() | dist_config.to_dict() | dist_metrics.to_dict() | pacb_config.to_dict() | pacb_metrics.to_dict()
    # metrics = asdict(base_config.hyperparams) | asdict(base_metrics) | asdict(dist_metrics) | asdict(pacb_metrics)
    df = pd.DataFrame([all_csv_values])
    df.to_csv(base_config.metrics_path, index=False)


def main():

    quick_test = True
    device = "cpu"
    dataset_name = "MNIST"
    seed = 0

    torch.manual_seed(seed)
    os.environ["WANDB_SILENT"] = "true"
    run = wandb.init()

    base_config = get_base_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        device=device
    )

    run.name = base_config.run_name
    run.save()

    print("Training base model...")
    init_model, base_model, base_metrics = train_base_model(base_config=base_config)
    init_model.save(base_config.model_init_dir, base_config.model_name)

    if base_metrics.reached_target:  # Only save if model reached target train loss
        print("Model reached target train loss")
        base_model.save(base_config.model_base_dir, base_config.model_name)
    else:
        print("Model failed to reach target train loss")
        run.finish()
        return

    dist_config = get_dist_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        use_whole_dataset=True,  # TODO: I think there should be a better place for this? Note it depends on the dataset.
        device=device,
        base_config=base_config,
        base_model=base_model,
    )

    print("Distilling model...")
    dist_model, dist_metrics = train_dist_model(
            base_model=base_model,
            dist_config=dist_config,
        )

    dist_model.save(base_config.model_dist_dir, base_config.model_name)

    pacb_config = get_pacb_config(quick_test=quick_test)
    print("Calculating PAC-Bayes bound...")
    pacb_metrics = get_pac_bound(
            init_model=init_model,
            base_model=base_model,
            base_config=base_config,
            pacb_config=pacb_config,
    )
    log_and_save_metrics(
        run_id=run.id,
        base_config=base_config,
        dist_config=dist_config,
        pacb_config=pacb_config,
        base_metrics=base_metrics,
        dist_metrics=dist_metrics,
        pacb_metrics=pacb_metrics,
    )
    run.finish()


if __name__ == "__main__":
    main()
