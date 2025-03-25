import os
import copy
import json
import pandas as pd

import torch
import wandb

from config import ExperimentConfig
from models import MLP
from load_data import get_dataloaders


toy_run = False

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"

dataset_name = "MNIST1D"
delta = 0.05
results_path = "sweep_results_2187_big_comb.csv"

if toy_run:
    train_size, test_size = 100, 100
    num_mc_samples_max_sigma = 10**2
    num_mc_samples_pac_bound = 10**2
    new_results_path = "sweep_results_2187_big_comb_copy.csv"
else:
    train_size, test_size = None, None
    num_mc_samples_max_sigma = 10**5
    num_mc_samples_pac_bound = 10**6
    new_results_path = "sweep_results_2187_big_comb.csv"


run = wandb.init()
wandb.run.name = f"op{wandb.config.optimizer_name}_hw{wandb.config.hidden_layer_width}_nl{wandb.config.num_hidden_layers}_lr{wandb.config.lr}_bs{wandb.config.batch_size}_dp{wandb.config.dropout_prob}_wd{wandb.config.weight_decay}"
wandb.run.save()

model_dims = [wandb.config.input_dim] + [wandb.config.hidden_layer_width] * wandb.config.num_hidden_layers + [wandb.config.output_dim]

base_experiment_config = ExperimentConfig(
    experiment="distillation",
    model_type="base",
    model_dims=model_dims,
    optimizer_name=wandb.config.optimizer_name,
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dropout_prob=wandb.config.dropout_prob,
    weight_decay=wandb.config.weight_decay,
    dataset_name=dataset_name,
    model_name=wandb.run.name,
)
init_experiment_config = ExperimentConfig(
    experiment="distillation",
    model_type="init",
    model_dims=model_dims,
    optimizer_name=wandb.config.optimizer_name,
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dropout_prob=wandb.config.dropout_prob,
    weight_decay=wandb.config.weight_decay,
    dataset_name=dataset_name,
    model_name=wandb.run.name,
)


def get_pac_bound():

    init_model = MLP(
        dimensions=base_experiment_config.model_dims,
        activation=base_experiment_config.model_act,
        dropout_prob=base_experiment_config.dropout_prob,
        device=device
    )
    base_model = copy.deepcopy(init_model)
    
    try:
        init_model.load(init_experiment_config.model_path)
    except FileNotFoundError:
        print(f"Warning: Init model not found at '{init_experiment_config.model_path}'")
        return
    try:
        base_model.load(base_experiment_config.model_path)
    except FileNotFoundError:
        print(f"Warning: Base model not found at '{base_experiment_config.model_path}'")
        return

    init_model.eval()
    base_model.eval()

    torch.manual_seed(0)
    train_loader, _ = get_dataloaders(
        base_experiment_config.dataset_name,
        base_experiment_config.batch_size,
        train_size=train_size,
        test_size=test_size,
    )

    max_sigma, noisy_error, sigmas_tried, errors, total_num_sigmas = base_model.get_max_sigma(dataset=train_loader.dataset, target_error_increase=0.1, num_mc_samples=num_mc_samples_max_sigma)
    noise_trials = [{"sigma": sigma, "noisy_error": error} for sigma, error in zip(sigmas_tried, errors)]
    pac_bound_inverse_kl = base_model.pac_bayes_error_bound_inverse_kl(prior=init_model, sigma=max_sigma, dataloader=train_loader, num_mc_samples=num_mc_samples_pac_bound, delta=delta, num_union_bounds=total_num_sigmas)
    pac_bound_pinsker = base_model.pac_bayes_error_bound_pinsker(prior=init_model, sigma=max_sigma, dataloader=train_loader, num_mc_samples=num_mc_samples_pac_bound, delta=delta, num_union_bounds=total_num_sigmas)
    
    print(f"{max_sigma=}, {noisy_error=}, {pac_bound_inverse_kl=}, {pac_bound_pinsker=}")
    print(f"{total_num_sigmas=}")

    results_df = pd.read_csv(results_path)
    if "max_sigma" not in results_df.columns:
        results_df["max_sigma"] = None
    if "noisy_error" not in results_df.columns:
        results_df["noisy_error"] = None
    if "noise_trials" not in results_df.columns:
        results_df["noise_trials"] = None
    if "pac_bound_inverse_kl" not in results_df.columns:
        results_df["pac_bound_inverse_kl"] = None
    if "pac_bound_pinsker" not in results_df.columns:
        results_df["pac_bound_pinsker"] = None
    
    row_indices = results_df.index[results_df["run_name"] == wandb.run.name].tolist()
    if not row_indices:
        print(f"Warning: Run name '{wandb.run.name}' not found in dataframe")
        return

    row_index = row_indices[0]
    if not results_df.at[row_index, "Reached Target Base"]:
        print(f"Not calculating PAC-Bayes bound as run name '{wandb.run.name}' did not reach target")
        return
    
    results_df.at[row_index, "max_sigma"] = max_sigma
    results_df.at[row_index, "noisy_error"] = noisy_error
    results_df.at[row_index, "noise_trials"] = json.dumps(noise_trials)
    results_df.at[row_index, "pac_bound_inverse_kl"] = pac_bound_inverse_kl
    results_df.at[row_index, "pac_bound_pinsker"] = pac_bound_pinsker

    results_df.to_csv(new_results_path, index=False)

    for sigma, noise in zip(sigmas_tried, errors):
        wandb.log({"sigma": sigma, "noise": noise})
    wandb.log({"Max sigma": max_sigma, "Final Noisy Error": noisy_error, "PAC Bound inverse kl": pac_bound_inverse_kl, "PAC Bound pinsker": pac_bound_pinsker})
    wandb.finish()


if __name__ == "__main__":

    os.makedirs("trained_models", exist_ok=True)
    print("Computing PAC-Bayes bound")
    get_pac_bound()