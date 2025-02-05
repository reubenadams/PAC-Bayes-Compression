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


dims = [(784, d, 10) for d in [100, 200, 300]]
batch_sizes = [32, 64, 128]
lrs = [0.01, 0.001, 0.0001]
target_overall_train_loss = 0.1
train_size, test_size = None, None
max_base_epochs = 100

configs = {
    (dim, batch_size, lr): Config(
        experiment="distillation",
        model_type="base",
        model_dims=dim,
        epochs=max_base_epochs,
        batch_size=batch_size,
        lr=lr,
    )
    for dim, batch_size, lr in product(dims, batch_sizes, lrs)
}


train_base_models, train_dist_models = True, True


if train_base_models:
    for (dim, batch_size, lr), config in configs.items():
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
        wandb.init(
            project="Distillation Base Full MNIST", name=f"{dim[1]}_{batch_size}_{lr}"
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
    wandb.init(project="Distillation Dist Full MNIST", name="Distillation")
    data = []
    for (dim, batch_size, lr), config in configs.items():
        print(dim, batch_size, lr)
        model_log = {
            "Dim": dim[1],
            "Batch Size": batch_size,
            "Learning Rate": lr,
        }
        model = MLP(config.model_dims, config.model_act, device=device)
        model.load(config.model_path)
        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            config.dataset,
            config.batch_size,
            train_size=train_size,
            test_size=test_size,
        )
        generalization_gap = model.get_generalization_gap(train_loader, test_loader)
        # generalization_gap = randint(1, 100)
        model_log["Generalization Gap"] = generalization_gap
        complexity = model.get_dist_complexity(
            dim_skip=1,
            max_hidden_dim=1000,
            dist_activation="relu",
            shift_logits=False,
            domain_train_loader=train_loader,
            lr=0.01,
            num_epochs=20,
            target_kl_on_train=0.01,
        )
        # complexity = randint(1, 100)
        if complexity:
            print(
                f"Successfully distilled model. Complexity: {complexity}, Generalization Gap: {generalization_gap}"
            )
            data.append([complexity, generalization_gap])
    table = wandb.Table(data=data, columns=["Complexity", "Generalization Gap"])
    wandb.log(
        {
            "my_custom_plot_id": wandb.plot.scatter(
                table,
                "Complexity",
                "Generalization Gap",
                title="Generalization Gap vs Complexity",
            )
        }
    )
    wandb.finish()
