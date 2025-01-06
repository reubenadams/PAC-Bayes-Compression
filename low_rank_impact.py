import os
from copy import deepcopy

import torch
import wandb

from models import MLP
from config import low_rank_config
from data.MNIST.load_data import train_loader, test_loader


wandb.init(project="Low rank impact", name=f"Model dims: {low_rank_config.model_dims}")

model = MLP(low_rank_config.model_dims, low_rank_config.model_act, low_rank=True)


try:
    model.load(low_rank_config.model_path)
    print(f"File {low_rank_config.model_path} found. Loading model...")
except FileNotFoundError:
    print(f"File {low_rank_config.model_path} not found. Training model...")
    base_train_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    base_test_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    base_optimizer = torch.optim.Adam(model.parameters(), lr=low_rank_config.learning_rate)
    model.train(
        train_loss_fn=base_train_loss_fn,
        test_loss_fn=base_test_loss_fn,
        optimizer=base_optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=low_rank_config.train_epochs,
        log_name="base_train_loss",
        get_accuracy=True
        )
    model.save(low_rank_config.model_path)





model.construct_low_rank_approxes()
for rank in range(2, 30):
    model.set_to_rank(rank)
    wandb.log({"rank": rank, "accuracy": model.overall_accuracy(test_loader).item()})
