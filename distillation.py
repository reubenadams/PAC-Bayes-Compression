import os

import torch
import wandb

from models import MLP
from config import full_mnist_config, dist_mnist_config
from load_data import (
    get_dataloaders,
    get_rand_domain_loader,
    get_mesh_domain_loader,
)


train_loader, test_loader = get_dataloaders(
    full_mnist_config.dataset,
    full_mnist_config.batch_size,
    # train_size=100,
    test_size=100,
    new_size=full_mnist_config.new_data_shape,
)

wandb.init(
    project="Distillation",
    name=f"Distilling into same architecture: {full_mnist_config.model_dims}",
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


full_model = MLP(
    full_mnist_config.model_dims, full_mnist_config.model_act, device=device
)
dist_data_model = MLP(
    full_mnist_config.model_dims, dist_mnist_config.model_act, device=device
)
os.makedirs(
    "trained_models/mnist/2x2", exist_ok=True
)  # TODO: Really need to sort this out with 2x2, 3x3, etc.!


try:

    full_model.load(full_mnist_config.model_path)
    print(f"File {full_mnist_config.model_path} found. Loading model...")

except FileNotFoundError:

    print(f"File {full_mnist_config.model_path} not found. Training model...")
    full_model.train(
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        lr=full_mnist_config.learning_rate,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=full_mnist_config.train_epochs,
        get_test_loss=True,
        get_test_accuracy=True,
        train_loss_name="Full Train Loss",
        test_loss_name="Full Test Loss",
        test_accuracy_name="Full Test Accuracy",
    )
    full_model.save(full_mnist_config.model_path)


try:

    dist_data_model.load(dist_mnist_config.model_path)
    print(f"File {dist_mnist_config.model_path} found. Loading model...")

except FileNotFoundError:

    print(f"File {dist_mnist_config.model_path} not found. Training model...")

    # domain_train_loader = get_rand_domain_loader(
    #     data_shape=dist_mnist_config.new_data_shape,
    #     sample_size=len(train_loader.dataset),
    #     batch_size=dist_mnist_config.batch_size,
    # )

    domain_train_loader = get_mesh_domain_loader(
        data_shape=dist_mnist_config.new_data_shape,
        epsilon=0.1,
    )

    domain_test_loader = get_mesh_domain_loader(
        data_shape=dist_mnist_config.new_data_shape,
        epsilon=0.1,
    )

    data_test_loader = test_loader

    dist_data_model.dist_from(
        full_model,
        domain_train_loader=domain_train_loader,
        domain_test_loader=domain_test_loader,
        data_test_loader=test_loader,
        lr=dist_mnist_config.learning_rate,
        num_epochs=dist_mnist_config.train_epochs,
        get_accuracy_on_test_data=True,
        get_kl_on_test_data=True,
        objective="l2",
        reduction="mellowmax",
    )
    dist_data_model.save(dist_mnist_config.model_path)


deviation = dist_data_model.max_l2_deviation(full_model, epsilon=1, data_shape=(2, 2))
print(f"Max deviation: {deviation}")
