import os
from copy import deepcopy
from itertools import product

import torch
import wandb

from models import LowRankMLP
from config import low_rank_config
from data.MNIST.load_data import train_loader, test_loader, get_B


# max_fro_norm = get_B(train_loader)
max_fro_norm = torch.tensor(28.)  # Reinstate previous line
print(f"Max Frobenius norm of training data: {max_fro_norm.item()}")

wandb.init(project="Low rank impact", name=f"Model dims: {low_rank_config.model_dims}")

model = LowRankMLP(low_rank_config.model_dims, low_rank_config.model_act)


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


for rank_comb in model.rank_combs:
    if model.valid_rank_combs[rank_comb]:
        print(f"rank_comb: {rank_comb}")
        eps = model.epsilons[rank_comb] * max_fro_norm
        num_params = model.num_params[rank_comb]
        min_UVs = model.min_UVs[rank_comb]
        max_UVs = model.max_UVs[rank_comb]
        min_Ss = model.min_Ss[rank_comb]
        max_Ss = model.max_Ss[rank_comb]
        model.set_to_ranks(rank_comb)
        margin_loss = model.margin_loss(test_loader, margin=torch.sqrt(torch.tensor(2.)) * eps)
        wandb.log({"ranks": rank_comb, "num_params": num_params, "min_UVs": min_UVs, "max_UVs": max_UVs, "min_Ss": min_Ss, "max_Ss": max_Ss, "eps": eps, "margin_loss": margin_loss})
