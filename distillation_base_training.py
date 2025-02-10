import os


import torch
import wandb
from itertools import product
import cProfile
import pstats


from config import TrainConfig, DistConfig, ExperimentConfig
from models import MLP
from load_data import get_dataloaders


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"

train_bases, train_dists = False, True

dataset_name = "MNIST1D"
# base_dims = [(40, d, 10) for d in [100, 200, 300, 400]]
# base_batch_sizes = [32, 64, 128]
# base_lrs = [0.01, 0.0032, 0.001]

base_dims = [(40, 100, 10)]
base_batch_sizes = [32]
base_lrs = [0.01]

train_size, test_size = None, None


base_train_configs = {}
base_experiment_configs = {}

for dims, batch_size, lr in product(base_dims, base_batch_sizes, base_lrs):
    base_train_configs[(batch_size, lr)] = TrainConfig(
        lr=lr,
        batch_size=batch_size,
        use_early_stopping=True,
        get_overall_train_loss=True,
    )
    base_experiment_configs[(dims, batch_size, lr)] = ExperimentConfig(
        project_name="Refactoring Distillation MNIST1D",
        experiment="distillation",
        model_type="base",
        model_dims=dims,
        lr=lr,
        batch_size=batch_size,
        dataset_name="MNIST1D",
    )

dist_config = DistConfig()


def train_base_models():
    for dims, batch_size, lr in product(base_dims, base_batch_sizes, base_lrs):
        train_config = base_train_configs[(batch_size, lr)]
        experiment_config = base_experiment_configs[(dims, batch_size, lr)]
        if train_config.log_with_wandb:
            wandb.init(
                project=experiment_config.project_name,
                name=f"{dims[1]}_{batch_size}_{lr}",
            )
        print(dims, batch_size, lr)
        torch.manual_seed(0)
        model = MLP(
            experiment_config.model_dims, experiment_config.model_act, device=device
        )
        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            experiment_config.dataset_name,
            experiment_config.batch_size,
            train_size=train_size,
            test_size=test_size,
        )
        overall_train_loss, target_loss_achieved = model.train(
            train_config,
            train_loader=train_loader,
            test_loader=test_loader,
            train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
            test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            overall_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        )
        if train_config.log_with_wandb:
            wandb.finish()
        if target_loss_achieved:  # Only save if model reached target train loss
            print(
                f"Model reached target train loss {overall_train_loss} <= {train_config.target_overall_train_loss}"
            )
            model.save(experiment_config.model_dir, experiment_config.model_name)
        else:
            print(
                f"Model did not reach target train loss {overall_train_loss} > {train_config.target_overall_train_loss}"
            )


def train_dist_models():

    for dims, batch_size, lr in product(base_dims, base_batch_sizes, base_lrs):
        base_experiment_config = base_experiment_configs[(dims, batch_size, lr)]

        if dist_config.log_with_wandb:
            wandb.finish()
            wandb.init(
                project=base_experiment_config.project_name,
                name=f"{dims[1]}_{batch_size}_{lr}",
                reinit=True,
            )

        print(dims, batch_size, lr)
        model_log = {
            "Dim": dims[1],
            "Batch Size": batch_size,
            "Learning Rate": lr,
        }
        model = MLP(
            base_experiment_config.model_dims,
            base_experiment_config.model_act,
            device=device,
        )

        try:
            model.load(base_experiment_config.model_path)
            print(f"File {base_experiment_config.model_path} found. Loading model...")
        except FileNotFoundError:
            print(
                f"File {base_experiment_config.model_path} not found. Skipping model..."
            )
            continue

        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            dataset_name=base_experiment_config.dataset_name,
            batch_size=dist_config.batch_size,
            train_size=train_size,
            test_size=test_size,
            use_whole_dataset=dist_config.use_whole_dataset,
            device=device,
        )

        # if dist_config.use_whole_dataset:
        #     train_loader = [(train_loader[0], train_loader[1])]
        #     test_loader = [(test_loader[0], test_loader[1])]

        generalization_gap = model.get_generalization_gap(train_loader, test_loader)
        model_log["Generalization Gap"] = generalization_gap

        complexity = model.get_dist_complexity(
            dist_config,
            domain_train_loader=train_loader,
        )

        if complexity:
            print(
                f"Successfully distilled model. Complexity: {complexity}, Generalization Gap: {generalization_gap}"
            )
            model_log["Complexity"] = complexity
            if dist_config.log_with_wandb:
                wandb.log(model_log)
        print()
        break
    if dist_config.log_with_wandb:
        wandb.finish()


if __name__ == "__main__":

    with cProfile.Profile() as pr:
        if train_bases:
            print("Training base")
            train_base_models()
        if train_dists:
            print("Training dist")
            train_dist_models()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(10)
