import os
import torch
import wandb
import copy
import pandas as pd
from typing import Optional

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


def train_and_save_base_model(
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

    try:

        print("Loading model and metrics...")
        init_model.load(base_config.model_init_dir, base_config.model_name)
        base_model.load(base_config.model_base_dir, base_config.model_name)
        base_metrics = config.BaseResults.load(base_config.base_metrics_path)
        print("Model and metrics loaded successfully.")
        init_model.eval()
        base_model.eval()
        return init_model, base_model, base_metrics
    
    except FileNotFoundError:
    
        print("Model or metrics not found, training from scratch.")

    base_metrics = base_model.train_model(
        base_config=base_config,
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        full_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
    )

    if base_metrics.reached_target:
        print("Base model reached target train loss")
        add_success(run_name=base_config.run_name, successful_runs_path=base_config.successful_runs_path)
    else:
        print("Base model did not reach target train loss")
    init_model.save(base_config.model_init_dir, base_config.model_name)
    base_model.save(base_config.model_base_dir, base_config.model_name)
    base_metrics.save(base_config.base_metrics_path)

    init_model.eval()
    base_model.eval()
    return init_model, base_model, base_metrics


def add_success(run_name: str, successful_runs_path: str) -> None:
    with open(successful_runs_path, "a") as f:
        f.write(f"{run_name}\n")


def train_and_save_dist_model(
        base_model: MLP,
        base_config: config.BaseConfig,
        dist_config: config.DistConfig,
    ) -> config.DistFinalResults:

    base_model.eval()

    try:

        print("Loading distillation metrics...")
        dist_metrics = config.DistFinalResults.load(base_config.dist_metrics_path)
        return dist_metrics
    
    except FileNotFoundError:

        print("Distillation metrics not found, distilling from scratch.")

    dist_model, dist_metrics = base_model.get_dist_complexity(dist_config=dist_config)
    dist_model.save(base_config.model_dist_dir, base_config.model_name)
    dist_metrics.save(base_config.dist_metrics_path)
    return dist_metrics


# We aren't minimizing over sigma because it's too expensive
def get_and_save_pac_bound(
        init_model: MLP,
        base_model: MLP,
        base_config: config.BaseConfig,
        pacb_config: config.PACBConfig,
        base_metrics: config.BaseResults,
) -> config.PACBResults:

    try:
        print("Loading PAC-Bayes metrics...")
        pacb_metrics = config.PACBResults.load(base_config.pacb_metrics_path)
        return pacb_metrics
    except FileNotFoundError:
        print("PAC-Bayes metrics not found, calculating from scratch.")

    init_model.eval()
    base_model.eval()

    sigma_target, noisy_CE_loss = base_model.get_sigma_target_CE_loss(
        dataset=base_config.data.train_loader.dataset,
        base_CE_loss=base_metrics.final_train_loss,
        CE_loss_increase=pacb_config.target_CE_loss_increase,
        num_mc_samples=pacb_config.num_mc_samples_sigma_target,
    )
    sigma_bound = base_model.get_sigma_rounded(sigma=sigma_target, sigma_tol=pacb_config.sigma_tol)
    if sigma_bound > pacb_config.sigma_max:
        print(f"Rounded sigma {sigma_bound} is greater than max sigma {pacb_config.sigma_max}.")
        sigma_bound = pacb_config.sigma_max

    pac_kl_bound = base_model.pacb_kl_bound(
        prior=init_model,
        sigma=sigma_bound,
        n=base_config.data.train_loader.dataset.__len__(),
        delta=pacb_config.delta,
        num_union_bounds=pacb_config.num_union_bounds,
    )
    pac_bound_inverse_kl = base_model.pacb_error_bound_inverse_kl(
        prior=init_model,
        sigma=sigma_bound,
        dataloader=base_config.data.train_loader,
        num_mc_samples=pacb_config.num_mc_samples_pac_bound,
        delta=pacb_config.delta,
        num_union_bounds=pacb_config.num_union_bounds,
        )
    pac_bound_pinsker = base_model.pacb_error_bound_pinsker(
        prior=init_model,
        sigma=sigma_bound,
        dataloader=base_config.data.train_loader,
        num_mc_samples=pacb_config.num_mc_samples_pac_bound,
        delta=pacb_config.delta,
        num_union_bounds=pacb_config.num_union_bounds,
        )
    
    print(f"{sigma_target=}, {sigma_bound=}, {noisy_CE_loss=}, {pac_kl_bound=}, {pac_bound_inverse_kl=}, {pac_bound_pinsker.item()=}")

    pacb_metrics = config.PACBResults(
        sigma_target=sigma_target.item(),
        sigma_bound=sigma_bound.item(),
        noisy_CE_loss=noisy_CE_loss.item(),
        kl_bound=pac_kl_bound.item(),
        error_bound_inverse_kl=pac_bound_inverse_kl.item(),
        error_bound_pinsker=pac_bound_pinsker.item(),
    )
    pacb_metrics.save(base_config.pacb_metrics_path)

    return pacb_metrics


@torch.no_grad()
def get_and_save_complexity_measures(
        init_model: MLP,
        base_model: MLP,
        base_config: config.BaseConfig,
        pacb_config: config.PACBConfig,
        base_metrics: config.BaseResults,
        dist_metrics: config.DistFinalResults,
        pacb_metrics: config.PACBResults,
):

    try:
        print("Loading complexity measures...")
        complexity_measures = config.ComplexityMeasures.load(base_config.complexity_measures_path)
        return complexity_measures
    except FileNotFoundError:
        print("Complexity measures not found, calculating from scratch.")

    complexity_measures = config.ComplexityMeasures(
        inverse_margin_tenth_percentile=base_model.get_inverse_margin_tenth_percentile(dataloader=base_config.data.train_loader).item(),
        train_loss=base_metrics.final_train_loss,
        train_error=1 - base_metrics.final_train_accuracy,
        output_entropy=base_model.get_avg_output_entropy(dataloader=base_config.data.train_loader).item(),

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
        
        inverse_squared_sigma_target=1 / (pacb_metrics.sigma_target ** 2),
        
        kl_bound_sigma_rounded=pacb_metrics.kl_bound,
        error_bound_sigma_rounded_inverse_kl=pacb_metrics.error_bound_inverse_kl,
        error_bound_sigma_rounded_pinsker=pacb_metrics.error_bound_pinsker,
        
        min_hidden_width=dist_metrics.complexity,

        target_CE_loss_increase=pacb_config.target_CE_loss_increase,
    )
    
    complexity_measures.save(base_config.complexity_measures_path)

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
    df.to_csv(base_config.all_metrics_path, index=False)


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

    init_model, base_model, base_metrics = train_and_save_base_model(base_config=base_config)

    if not base_metrics.reached_target:
        # If the model did not reach the target train loss log the metrics and finish the run early
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

    dist_metrics = train_and_save_dist_model(
            base_model=base_model,
            base_config=base_config,
            dist_config=dist_config,
        )

    pacb_config = get_pacb_config(quick_test=quick_test)
    pacb_metrics = get_and_save_pac_bound(
            init_model=init_model,
            base_model=base_model,
            base_config=base_config,
            pacb_config=pacb_config,
            base_metrics=base_metrics,
    )

    complexity_measures = get_and_save_complexity_measures(
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
