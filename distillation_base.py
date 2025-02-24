import os

import torch
import wandb

from config import TrainConfig, ExperimentConfig
from models import MLP
from load_data import get_dataloaders


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"

dataset_name = "MNIST1D"
train_size, test_size = None, None
num_epochs = 2000

run = wandb.init()
wandb.run.name = f"hw{wandb.config.dims[1]}_lr{wandb.config.lr}_bs{wandb.config.batch_size}_dp{wandb.config.dropout_prob}"
wandb.run.save()


base_train_config = TrainConfig(
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dropout_prob=wandb.config.dropout_prob,
    num_epochs=num_epochs,
    use_early_stopping=True,
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
    model_dims=wandb.config.dims,
    lr=wandb.config.lr,
    batch_size=wandb.config.batch_size,
    dropout_prob=wandb.config.dropout_prob,
    dataset_name=dataset_name,
)


def train_base_models():

    model = MLP(
        base_experiment_config.model_dims, base_experiment_config.model_act, device=device
    )

    torch.manual_seed(0)
    train_loader, test_loader = get_dataloaders(
        base_experiment_config.dataset_name,
        base_experiment_config.batch_size,
        train_size=train_size,
        test_size=test_size,
    )

    overall_train_loss, target_loss_achieved = model.train(
        base_train_config,
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