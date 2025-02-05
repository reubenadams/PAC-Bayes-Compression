import torch
import wandb
from itertools import product


from config import Config
from models import MLP
from load_data import get_dataloaders


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


dims = [[784, d, 10] for d in [100, 200, 300]]
batch_sizes = [32, 64, 128]
lrs = [0.01, 0.001, 0.0001]
target_overall_train_loss = 0.01


for dim, batch_size, lr in product(dims, batch_sizes, lrs):
    print(dim, batch_size, lr)
    config = Config(
        experiment="distillation",
        model_type="base",
        model_dims=dim,
        epochs=1000,
        batch_size=batch_size,
        lr=lr,
    )
    model = MLP(config.model_dims, config.model_act, device=device)
    train_loader, test_loader = get_dataloaders(
        config.dataset,
        config.batch_size,
        train_size=100,
        test_size=100,
    )
    wandb.init(project="Distillation Base", name=f"{dim[1]}_{batch_size}_{lr}")
    overall_train_loss = model.train(
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
    if overall_train_loss:  # Only save if model reached target train loss
        if overall_train_loss <= target_overall_train_loss:
            model.save(config.model_dir, config.model_name)
