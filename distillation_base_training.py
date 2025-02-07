import os


import torch
import wandb
from itertools import product



from config import Config
from models import MLP
from load_data import get_dataloaders


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"


dims = [(40, d, 10) for d in [100, 200, 300, 400]]
batch_sizes = [32, 64, 128]
lrs = [0.01, 0.0032, 0.001]
target_overall_train_loss = 0.01
target_kl_loss = 0.01
train_size, test_size = None, None
max_base_epochs = 2000
max_dist_epochs = 2000
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
    for dim, batch_size, lr in product(dims, batch_sizes, lrs)
}


train_base_models, train_dist_models = False, True


if train_base_models:
    for (dim, batch_size, lr), config in configs.items():
        wandb.init(
            project="Distillation Base MNIST1D",
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
        )
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


if train_dist_models:
    for (dim, batch_size, lr), config in configs.items():

        wandb.init(
            project="Distillation Dist MNIST1D",
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
            config.dataset,
            config.batch_size,
            train_size=train_size,
            test_size=test_size,
        )

        generalization_gap = model.get_generalization_gap(train_loader, test_loader)
        model_log["Generalization Gap"] = generalization_gap

        complexity = model.get_dist_complexity(
            dim_skip=dist_dim_skip,
            max_hidden_dim=1000,
            dist_activation="relu",
            shift_logits=False,
            domain_train_loader=train_loader,
            lr=0.001,
            num_epochs=max_dist_epochs,
            target_kl_on_train=target_kl_loss,
            patience=dist_patience,
        )

        if complexity:
            print(
                f"Successfully distilled model. Complexity: {complexity}, Generalization Gap: {generalization_gap}"
            )
            model_log["Complexity"] = complexity
            wandb.log(model_log)
        print()

    wandb.finish()
