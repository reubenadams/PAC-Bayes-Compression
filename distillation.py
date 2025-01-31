import os

import torch
import wandb

from models import MLP
from config import full_mnist_config, dist_data_mnist_config
from load_data import get_dataloaders, get_rand_domain_dataloader


train_loader, test_loader = get_dataloaders(
    full_mnist_config.dataset,
    full_mnist_config.batch_size,
    # train_size=100,
    # test_size=100,
    new_size=full_mnist_config.new_size,
)

wandb.init(
    project="Distillation",
    name=f"Distilling into same architecture: {full_mnist_config.model_dims}",
)

full_model = MLP(full_mnist_config.model_dims, full_mnist_config.model_act)
dist_data_model = MLP(full_mnist_config.model_dims, dist_data_mnist_config.model_act)
os.makedirs(
    "trained_models/mnist/2x2", exist_ok=True
)  # TODO: Really need to sort this out with 2x2, 3x3, etc.!


try:
    full_model.load(full_mnist_config.model_path)
    print(f"File {full_mnist_config.model_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {full_mnist_config.model_path} not found. Training model...")
    full_train_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    full_test_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    full_optimizer = torch.optim.Adam(
        full_model.parameters(), lr=full_mnist_config.learning_rate
    )
    full_model.train(
        train_loss_fn=full_train_loss_fn,
        test_loss_fn=full_test_loss_fn,
        optimizer=full_optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=full_mnist_config.train_epochs,
        log_name="full_train_loss",
        get_accuracy=True,
    )
    full_model.save(full_mnist_config.model_path)


try:
    dist_data_model.load(dist_data_mnist_config.model_path)
    print(f"File {dist_data_mnist_config.model_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {dist_data_mnist_config.model_path} not found. Training model...")
    dist_optimizer = torch.optim.Adam(
        dist_data_model.parameters(), lr=dist_data_mnist_config.learning_rate
    )
    dist_data_model.dist_from(
        full_model,
        optimizer=dist_optimizer,
        # train_loader=train_loader,
        train_loader=get_rand_domain_dataloader(
            data_size=dist_data_mnist_config.new_size,
            sample_size=len(train_loader.dataset),
            batch_size=dist_data_mnist_config.batch_size,
        ),
        test_loader=test_loader,
        num_epochs=dist_data_mnist_config.train_epochs,
        log_name="dist_train_loss",
        get_accuracy=True,
        objective="l2",
        reduction="mellowmax",
    )
    dist_data_model.save(dist_data_mnist_config.model_path)


deviation = dist_data_model.max_l2_deviation(full_model, epsilon=1, data_size=(2, 2))
print(f"Max deviation: {deviation}")
