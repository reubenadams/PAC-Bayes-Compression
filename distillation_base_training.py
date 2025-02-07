import os


import torch
import wandb
from itertools import product
import cProfile
import pstats


from config import Config
from models import MLP
from load_data import get_dataloaders


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"


dims = [(40, d, 10) for d in [100, 200, 300, 400]]
base_batch_sizes = [32, 64, 128]
lrs = [0.01, 0.0032, 0.001]
target_overall_train_loss = 0.01
target_kl_loss = 0.01
dist_batch_size = 128
max_hidden_dim = 7
train_size, test_size = None, None
max_base_epochs = 2000
max_dist_epochs = 100
base_patience = 20
dist_patience = 20
dist_dim_skip = 15

configs = {
    (dim, batch_size, lr): Config(
        experiment="distillation",
        model_type="base",
        model_dims=dim,
        dataset="MNIST1D",
        epochs=max_base_epochs,
        batch_size=batch_size,
        lr=lr,
    )
    for dim, batch_size, lr in product(dims, base_batch_sizes, lrs)
}


train_base_models, train_dist_models = False, True
log = False


if train_base_models:
    for (dim, batch_size, lr), config in configs.items():
        if log:
            wandb.init(
                project="Accelerating Distillation MNIST1D",
                name=f"{dim[1]}_{batch_size}_{lr}",
            )
        print(dim, batch_size, lr)
        torch.manual_seed(0)
        model = MLP(config.model_dims, config.model_act, device=device)
        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            config.dataset,
            config.batch_size,
            train_size=train_size,
            test_size=test_size,
        )
        overall_train_loss, target_loss_achieved = model.train(
            train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
            test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            lr=config.lr,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config.epochs,
            get_overall_train_loss=True,
            overall_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            get_test_loss=False,
            get_test_accuracy=False,
            train_loss_name="Train Loss",
            test_loss_name="Test Loss",
            test_accuracy_name="Test Accuracy",
            target_overall_train_loss=target_overall_train_loss,
            patience=base_patience,
            log_with_wandb=log,
        )
        if log:
            wandb.finish()
        if target_loss_achieved:  # Only save if model reached target train loss
            print(
                f"Model reached target train loss {overall_train_loss} <= {target_overall_train_loss}"
            )
            model.save(config.model_dir, config.model_name)
        else:
            print(
                f"Model did not reach target train loss {overall_train_loss} > {target_overall_train_loss}"
            )

        break


# if train_dist_models:
def train_dist_models():
    for (dim, batch_size, lr), config in configs.items():

        if log:
            wandb.init(
                project="Accelerating Distillation MNIST1D",
                name=f"{dim[1]}_{batch_size}_{lr}",
                reinit=True,
            )

        print(dim, batch_size, lr)
        model_log = {
            "Dim": dim[1],
            "Batch Size": batch_size,
            "Learning Rate": lr,
        }
        model = MLP(config.model_dims, config.model_act, device=device)

        try:
            model.load(config.model_path)
            print(f"File {config.model_path} found. Loading model...")
        except FileNotFoundError:
            print(f"File {config.model_path} not found. Skipping model...")
            continue

        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            dataset_name=config.dataset,
            batch_size=dist_batch_size,
            train_size=train_size,
            test_size=test_size,
            device=device,
        )

        generalization_gap = model.get_generalization_gap(train_loader, test_loader)
        model_log["Generalization Gap"] = generalization_gap

        complexity = model.get_dist_complexity(
            dim_skip=dist_dim_skip,
            max_hidden_dim=max_hidden_dim,
            dist_activation="relu",
            shift_logits=False,
            domain_train_loader=train_loader,
            lr=0.001,
            batch_size=dist_batch_size,
            num_epochs=max_dist_epochs,
            target_kl_on_train=target_kl_loss,
            patience=dist_patience,
            log_with_wandb=log,
        )

        if complexity:
            print(
                f"Successfully distilled model. Complexity: {complexity}, Generalization Gap: {generalization_gap}"
            )
            model_log["Complexity"] = complexity
            if log:
                wandb.log(model_log)
        print()
        break
    if log:
        wandb.finish()


with cProfile.Profile() as pr:
    train_dist_models()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME).print_stats(30)
