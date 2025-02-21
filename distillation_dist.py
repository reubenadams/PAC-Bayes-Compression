import os

import torch
import wandb

from config import DistTrainConfig, ExperimentConfig
from models import MLP
from load_data import get_dataloaders


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"

dataset_name = "MNIST1D"
train_size, test_size = None, None


num_dist_attempts = 5

run = wandb.init(reinit=True)
wandb.run.name = f"hw{wandb.config.dims[1]}_lr{wandb.config.lr}_bs{wandb.config.batch_size}"
wandb.run.save()


base_experiment_config = ExperimentConfig(
    project_name=f"Distillation {dataset_name} Base",  # This isn't used when running a wandb sweep
    experiment="distillation",
    model_type="base",
    model_dims=wandb.config.dims,
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dataset_name=dataset_name,
)
dist_experiment_config = ExperimentConfig(
    project_name=f"Distillation {dataset_name} Dist, Binary Search",  # This isn't used when running a wandb sweep
    experiment="distillation",
    model_type="dist",
    model_dims=wandb.config.dims,
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dataset_name=dataset_name,
)

dist_train_config = DistTrainConfig(use_whole_dataset=True, use_early_stopping=True)


def train_dist_models():

    if os.path.isfile(base_experiment_config.model_path):

        print(wandb.config.dims, wandb.config.batch_size, wandb.config.lr)
        model_log = {
            "Dim": wandb.config.dims[1],
            "Batch Size": wandb.config.batch_size,
            "Learning Rate": wandb.config.lr,
        }
        base_model = MLP(
            base_experiment_config.model_dims,
            base_experiment_config.model_act,
            device=device,
        )

        base_model.load(base_experiment_config.model_path)

        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            dataset_name=base_experiment_config.dataset_name,
            batch_size=dist_train_config.batch_size,
            train_size=train_size,
            test_size=test_size,
            use_whole_dataset=dist_train_config.use_whole_dataset,
            device=device,
        )

        generalization_gap = base_model.get_generalization_gap(
            train_loader, test_loader
        )
        model_log["Generalization Gap"] = generalization_gap

        complexity = base_model.get_dist_complexity(
            dist_train_config,
            domain_train_loader=train_loader,
            num_attempts=num_dist_attempts,
        )
        model_log["Complexity"] = complexity

        if dist_train_config.log_with_wandb:
            wandb.log(model_log)

        print()

        if dist_train_config.log_with_wandb:
            wandb.finish()


if __name__ == "__main__":

    os.makedirs("trained_models", exist_ok=True)
    print("Training dist")
    train_dist_models()
