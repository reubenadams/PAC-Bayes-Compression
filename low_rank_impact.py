import os
from copy import deepcopy

import torch
import wandb

from models import LowRankMLP
from config import low_rank_CIFAR10_config as model_config
from load_data import get_dataloaders, get_B


train_loader, test_loader = get_dataloaders(
    model_config.dataset,
    model_config.batch_size,
    train_size=100,
    test_size=100,
)

max_fro_norm = get_B(train_loader)
# max_fro_norm = torch.tensor(28.)  # Reinstate previous line
print(f"Max Frobenius norm of training data: {max_fro_norm.item()}")

wandb.init(
    project="Margin loss against margin, CIFAR",
    name=f"Model dims: {model_config.model_dims}",
)

model = LowRankMLP(model_config.model_dims, model_config.model_act)


try:

    model.load(model_config.model_path)
    print(f"File {model_config.model_path} found. Loading model...")

except FileNotFoundError:

    print(f"File {model_config.model_path} not found. Training model...")
    base_train_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    base_test_loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    model.train(
        train_loss_fn=base_train_loss_fn,
        test_loss_fn=base_test_loss_fn,
        lr=model_config.lr,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=model_config.epochs,
        get_test_loss=True,
        get_test_accuracy=True,
    )
    model.save(model_config.model_dir, model_config.model_name)


# Log margin loss of full rank model:
for margin in torch.linspace(0, 25, 251):
    margin_loss_logits = model.overall_margin_loss(
        test_loader, margin, take_softmax=False
    )
    wandb.log({"Margin": margin, "Margin loss logits": margin_loss_logits})
    if margin <= 1:
        margin_loss_probs = model.overall_margin_loss(
            test_loader, margin, take_softmax=True
        )
        wandb.log({"Margin": margin, "Margin loss probs": margin_loss_probs})


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
        margin_loss = model.overall_margin_loss(
            test_loader, margin=torch.sqrt(torch.tensor(2.0)) * eps, take_softmax=True
        )
        wandb.log(
            {
                "ranks": rank_comb,
                "num_params": num_params,
                "min_UVs": min_UVs,
                "max_UVs": max_UVs,
                "min_Ss": min_Ss,
                "max_Ss": max_Ss,
                "eps": eps,
                "margin_loss": margin_loss,
            }
        )
