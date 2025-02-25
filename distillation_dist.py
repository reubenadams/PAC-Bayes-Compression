import os

import torch
import wandb

from config import DistTrainConfig, ExperimentConfig
from models import MLP
from load_data import get_dataloaders


toy_run = False

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"

dataset_name = "MNIST1D"

if toy_run:
    train_size, test_size = 100, 100
    max_epochs = 10000
    num_dist_attempts = 1
    target_kl_on_train = 0.1
    patience = 10
else:
    train_size, test_size = None, None
    max_epochs = 100000
    num_dist_attempts = 5
    target_kl_on_train = 0.01
    patience = 100


run = wandb.init(reinit=True)
wandb.run.name = f"op{wandb.config.optimizer_name}_hw{wandb.config.hidden_layer_width}_nl{wandb.config.num_hidden_layers}_lr{wandb.config.lr}_bs{wandb.config.batch_size}_dp{wandb.config.dropout_prob}_wd{wandb.config.weight_decay}"
wandb.run.save()

model_dims = [wandb.config.input_dim] + [wandb.config.hidden_layer_width] * wandb.config.num_hidden_layers + [wandb.config.output_dim]

base_experiment_config = ExperimentConfig(
    project_name=f"Distillation {dataset_name} Base",  # This isn't used when running a wandb sweep
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
dist_experiment_config = ExperimentConfig(
    project_name=f"Distillation {dataset_name} Dist, Binary Search",  # This isn't used when running a wandb sweep
    experiment="distillation",
    model_type="dist",
    model_dims=model_dims,
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dropout_prob=wandb.config.dropout_prob,
    weight_decay=wandb.config.weight_decay,
    dataset_name=dataset_name,
    model_name=wandb.run.name,
)

dist_train_config = DistTrainConfig(
    max_epochs=max_epochs,
    use_whole_dataset=True,
    use_early_stopping=True,
    target_kl_on_train=target_kl_on_train,
    patience=patience,
)


def train_dist_models():

    if os.path.isfile(base_experiment_config.model_path):

        print(model_dims, wandb.config.batch_size, wandb.config.lr)
        model_log = {
            "Optimizer": wandb.config.optimizer_name,
            "Width": wandb.config.hidden_layer_width,
            "Depth": wandb.config.num_hidden_layers,
            "Batch Size": wandb.config.batch_size,
            "Learning Rate": wandb.config.lr,
            "Dropout Probability": wandb.config.dropout_prob,
            "Weight Decay": wandb.config.weight_decay,
        }
        base_model = MLP(
            dimensions=base_experiment_config.model_dims,
            activation=base_experiment_config.model_act,
            dropout_prob=base_experiment_config.dropout_prob,
            device=device,
        )
        base_model.eval()

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

        complexity, dist_model = base_model.get_dist_complexity(
            dist_train_config,
            domain_train_loader=train_loader,
            num_attempts=num_dist_attempts,
        )
        model_log["Complexity"] = complexity

        dist_model.save(dist_experiment_config.model_dir, dist_experiment_config.model_name)

        if dist_train_config.log_with_wandb:
            wandb.log(model_log)

        print()

        if dist_train_config.log_with_wandb:
            wandb.finish()


if __name__ == "__main__":

    os.makedirs("trained_models", exist_ok=True)
    print("Training dist")
    train_dist_models()
