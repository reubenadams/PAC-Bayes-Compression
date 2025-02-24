import os

import torch
import wandb

from config import TrainConfig, ExperimentConfig
from models import MLP
from load_data import get_dataloaders


toy_run = True

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"

dataset_name = "MNIST1D"

if toy_run:
    train_size, test_size = 100, 100
    num_epochs = 200
    patience = 10
    target_overall_train_loss = 0.1
else:
    train_size, test_size = None, None
    num_epochs = 2000
    patience = 50
    target_overall_train_loss = 0.01


run = wandb.init()
wandb.run.name = f"hw{wandb.config.model_width}_nl{wandb.config.model_depth}_lr{wandb.config.lr}_bs{wandb.config.batch_size}_dp{wandb.config.dropout_prob}_wd{wandb.config.weight_decay}"
wandb.run.save()

model_dims = [wandb.config.input_dim] + [wandb.config.model_width] * wandb.config.model_depth + [wandb.config.output_dim]

base_train_config = TrainConfig(
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dropout_prob=wandb.config.dropout_prob,
    weight_decay=wandb.config.weight_decay,
    num_epochs=num_epochs,
    use_early_stopping=True,
    target_overall_train_loss=target_overall_train_loss,
    patience=patience,
    get_overall_train_loss=True,
    get_test_accuracy=True,
    train_loss_name="Base Train Loss",
    test_loss_name="Base Test Loss",
    test_accuracy_name="Base Test Accuracy",
)
base_experiment_config = ExperimentConfig(
    project_name=f"Distillation {dataset_name} Base",  # This isn't used when running a wandb sweep
    experiment="distillation",
    model_type="base",
    model_dims=model_dims,
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dropout_prob=wandb.config.dropout_prob,
    weight_decay=wandb.config.weight_decay,
    dataset_name=dataset_name,
)


def train_base_models():

    model = MLP(
        dimensions=base_experiment_config.model_dims,
        activation=base_experiment_config.model_act,
        dropout_prob=base_experiment_config.dropout_prob,
        device=device
    )

    torch.manual_seed(0)
    train_loader, test_loader = get_dataloaders(
        base_experiment_config.dataset_name,
        base_experiment_config.batch_size,
        train_size=train_size,
        test_size=test_size,
    )

    overall_train_loss, target_loss_achieved = model.train_model(
        train_config=base_train_config,
        train_loader=train_loader,
        test_loader=test_loader,
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        overall_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
    )

    if target_loss_achieved:  # Only save if model reached target train loss
        print(
            f"Model reached target train loss {overall_train_loss} <= {base_train_config.target_overall_train_loss}"
        )
        model.save(base_experiment_config.model_dir, base_experiment_config.model_name)
    else:
        print(
            f"Model did not reach target train loss {overall_train_loss} > {base_train_config.target_overall_train_loss}"
        )

    if base_train_config.log_with_wandb:
        wandb.finish()

if __name__ == "__main__":

    os.makedirs("trained_models", exist_ok=True)
    print("Training base")
    train_base_models()