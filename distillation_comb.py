import os
import torch
import wandb
import copy
import pandas as pd

import config
from models import MLP


def get_base_config(quick_test: bool, dataset_name: str, device: str, experiment_type: str):
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
        experiment_type=experiment_type,
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
        train_dataset=base_config.data.train_loader.dataset,
        test_dataset=base_config.data.test_loader.dataset,
        data_filepath=base_config.data.data_filepath,
    )
    data_config.add_base_logit_loaders(
        base_model=base_model,
        train_dataset=base_config.data.train_loader.dataset,
        test_dataset=base_config.data.test_loader.dataset,
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
        activation_name=base_config.hyperparams.activation_name,
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
    pac_bound_inverse_kl = base_model.pacb_error_bound_inverse_kl(
        prior=init_model,
        sigma=max_sigma,
        dataloader=base_config.data.train_loader,
        num_mc_samples=pacb_config.num_mc_samples_pac_bound,
        delta=pacb_config.delta,
        num_union_bounds=total_num_sigmas,
        )
    pac_bound_pinsker = base_model.pacb_error_bound_pinsker(
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
        pac_bound_inverse_kl=pac_bound_inverse_kl.item(),
        pac_bound_pinsker=pac_bound_pinsker.item(),
    )

    return pacb_metrics


@torch.no_grad()
def get_complexity_measures(
        init_model: MLP,
        base_model: MLP,
        base_config: config.BaseConfig,
        pacb_config: config.PACBConfig,
        base_metrics: config.BaseResults,
        dist_metrics: config.DistFinalResults,
        pacb_metrics: config.PACBResults,
):
    inverse_margin_tenth_percentile = base_model.get_inverse_margin_tenth_percentile(dataloader=base_config.data.train_loader).item()
    train_loss = base_metrics.final_train_loss
    output_entropy = base_model.get_avg_output_entropy(dataloader=base_config.data.train_loader).item()

    sigma_ten_percent_increase = base_model.get_sigma_ten_percent_increase(
        dataloader=base_config.data.train_loader,
        base_error=1 - base_metrics.final_train_accuracy,
        num_mc_samples=pacb_config.num_mc_samples_max_sigma,
    ).item()
    inverse_squared_sigma_ten_percent_increase = 1 / (sigma_ten_percent_increase ** 2)

    kl_bound_sigma_ten_percent_increase = base_model.pacb_kl_bound(
        prior=init_model,
        sigma=sigma_ten_percent_increase,
        n=len(base_config.data.train_loader.dataset),
        delta=pacb_config.delta,
        num_union_bounds=1,
    ).item()

    error_bound_min_over_sigma_inverse_kl = pacb_metrics.pac_bound_inverse_kl
    error_bound_min_over_sigma_pinsker = pacb_metrics.pac_bound_pinsker

    min_hidden_width = dist_metrics.complexity

    complexity_measures = config.ComplexityMeasures(
        inverse_margin_tenth_percentile=inverse_margin_tenth_percentile,
        train_loss=train_loss,
        train_error=1 - base_metrics.final_train_accuracy,
        output_entropy=output_entropy,
        l1_norm=base_model.mu_l1_norm().item(),
        l2_norm=base_model.mu_l2_norm().item(),
        l1_norm_from_init=base_model.mu_l1_norm_from_init(other=init_model).item(),
        l2_norm_from_init=base_model.mu_l2_norm_from_init(other=init_model).item(),
        spectral_sum=base_model.mu_spectral_sum().item(),
        spectral_product=base_model.mu_spectral_product().item(),
        frobenius_sum=base_model.mu_frobenius_sum().item(),
        frobenius_product=base_model.mu_frobenius_product().item(),
        spectral_sum_from_init=base_model.mu_spectral_sum_from_init(other=init_model).item(),
        spectral_product_from_init=base_model.mu_spectral_product_from_init(other=init_model).item(),
        frobenius_sum_from_init=base_model.mu_frobenius_sum_from_init(other=init_model).item(),
        frobenius_product_from_init=base_model.mu_frobenius_product_from_init(other=init_model).item(),
        inverse_squared_sigma_ten_percent_increase=inverse_squared_sigma_ten_percent_increase,
        kl_bound_sigma_ten_percent_increase=kl_bound_sigma_ten_percent_increase,
        error_bound_min_over_sigma_inverse_kl=error_bound_min_over_sigma_inverse_kl,
        error_bound_min_over_sigma_pinsker=error_bound_min_over_sigma_pinsker,
        min_hidden_width=min_hidden_width,
    )
    
    return complexity_measures


def log_and_save_metrics(
        run_id: str,
        base_config: config.BaseConfig,
        dist_config: config.DistConfig = None,
        pacb_config: config.PACBConfig = None,
        base_metrics: config.BaseResults = None,
        dist_metrics: config.DistFinalResults = None, 
        pacb_metrics: config.PACBResults = None,
        complexity_measures: config.ComplexityMeasures = None,
) -> None:
    all_configs = {"Run ID": run_id, "Run Name": base_config.run_name} | base_config.to_dict()
    if dist_config is not None:
        all_configs |= dist_config.to_dict()
    if pacb_config is not None:
        all_configs |= pacb_config.to_dict()

    base_metrics.log()
    all_metrics = base_metrics.to_dict()
    if dist_metrics is not None:
        dist_metrics.log()
        all_metrics |= dist_metrics.to_dict()
    if pacb_metrics is not None:
        pacb_metrics.log()
        all_metrics |= pacb_metrics.to_dict()
    if complexity_measures is not None:
        complexity_measures.log()
        all_metrics |= complexity_measures.to_dict()

    df = pd.DataFrame([all_configs | all_metrics])
    df.to_csv(base_config.dist_metrics_path, index=False)


def main():

    quick_test = False
    device = "cpu"
    dataset_name = "MNIST1D"
    seed = 0

    torch.manual_seed(seed)
    os.environ["WANDB_SILENT"] = "true"
    run = wandb.init()

    base_config = get_base_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        device=device,
        experiment_type="distillation",
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
        # If the model did not reach the target train loss log the metrics and finish the run early
        print("Model failed to reach target train loss")
        log_and_save_metrics(
            run_id=run.id,
            base_config=base_config,
            base_metrics=base_metrics,
        )
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

    print("Calculating complexity measures...")
    complexity_measures = get_complexity_measures(
        init_model=init_model,
        base_model=base_model,
        base_config=base_config,
        pacb_config=pacb_config,
        base_metrics=base_metrics,
        dist_metrics=dist_metrics,
        pacb_metrics=pacb_metrics,
    )

    print("Logging and saving metrics...")
    log_and_save_metrics(
        run_id=run.id,
        base_config=base_config,
        dist_config=dist_config,
        pacb_config=pacb_config,
        base_metrics=base_metrics,
        dist_metrics=dist_metrics,
        pacb_metrics=pacb_metrics,
        complexity_measures=complexity_measures,
    )
    run.finish()


if __name__ == "__main__":
    main()
