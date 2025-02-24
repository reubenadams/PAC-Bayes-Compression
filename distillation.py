import torch
import wandb

from models import MLP
from config import full_mnist_config, dist_kl_mnist_config, dist_l2_mnist_config
from load_data import (
    get_dataloaders,
    get_rand_domain_loader,
    get_mesh_domain_loader,
)


torch.manual_seed(0)


dist_kl_mnist_config.model_name = dist_kl_mnist_config.model_name[:-2] + "_kl.t"
dist_l2_mnist_config.model_name = dist_l2_mnist_config.model_name[:-2] + "_l2.t"

dist_kl_mnist_config.model_path = dist_kl_mnist_config.model_path[:-2] + "_kl.t"
dist_l2_mnist_config.model_path = dist_l2_mnist_config.model_path[:-2] + "_l2.t"


train_loader, test_loader = get_dataloaders(
    full_mnist_config.dataset_name,
    full_mnist_config.batch_size,
    train_size=100,
    test_size=100,
    new_input_size=full_mnist_config.new_data_shape,
)

wandb.init(
    project="Distillation",
    name=f"No logit shift, alpha=0.1",
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(0)
full_model = MLP(
    dimensions=full_mnist_config.model_dims,
    activation=full_mnist_config.model_act,
    device=device,
    shift_logits=True,
)
torch.manual_seed(0)
dist_kl_data_model = MLP(
    dimensions=full_mnist_config.model_dims,
    activation=dist_kl_mnist_config.model_act,
    device=device,
    shift_logits=True,
)
torch.manual_seed(0)
dist_l2_data_model = MLP(
    dimensions=full_mnist_config.model_dims,
    activation=dist_l2_mnist_config.model_act,
    device=device,
    shift_logits=True,
)


# Train the full model
try:

    full_model.load(full_mnist_config.model_path)
    print(f"File {full_mnist_config.model_path} found. Loading model...")

except FileNotFoundError:

    print(f"File {full_mnist_config.model_path} not found. Training model...")
    full_model.train_model(
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        lr=full_mnist_config.lr,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=full_mnist_config.epochs,
        get_test_accuracy=True,
        train_loss_name="Full Train Loss",
        test_loss_name="Full Test Loss",
        test_accuracy_name="Full Test Accuracy",
    )
    full_model.save(full_mnist_config.model_dir, full_mnist_config.model_name)


mesh_loader = get_mesh_domain_loader(
    data_shape=dist_kl_mnist_config.new_data_shape, epsilon=0.05
)


# Train the distillation model with KL divergence
try:

    dist_kl_data_model.load(dist_kl_mnist_config.model_path)
    print(f"File {dist_kl_mnist_config.model_path} found. Loading model...")

except FileNotFoundError:

    data_test_loader = test_loader
    torch.manual_seed(0)
    dist_kl_data_model.dist_from(
        full_model,
        domain_train_loader=mesh_loader,
        domain_test_loader=mesh_loader,
        data_test_loader=test_loader,
        lr=dist_kl_mnist_config.lr,
        num_epochs=dist_kl_mnist_config.epochs,
        get_accuracy_on_test_data=True,
        get_kl_on_test_data=True,
        objective="kl",
        reduction="mean",
    )
    dist_kl_data_model.save(
        dist_kl_mnist_config.model_dir, dist_kl_mnist_config.model_name
    )


# Train the distillation model with max L2
try:

    dist_l2_data_model.load(dist_l2_mnist_config.model_path)
    print(f"File {dist_l2_mnist_config.model_path} found. Loading model...")

except FileNotFoundError:

    print(f"File {dist_l2_mnist_config.model_path} not found. Training model...")
    dist_l2_data_model.load(dist_kl_mnist_config.model_path)
    torch.manual_seed(0)
    dist_l2_data_model.dist_from(
        full_model,
        domain_train_loader=mesh_loader,
        domain_test_loader=mesh_loader,
        data_test_loader=test_loader,
        lr=dist_l2_mnist_config.lr,
        num_epochs=dist_l2_mnist_config.epochs,
        epoch_shift=dist_kl_mnist_config.epochs,
        get_accuracy_on_test_data=True,
        get_kl_on_test_data=True,
        objective="l2",
        reduction="mellowmax",
        alpha=0.1,
    )
    dist_l2_data_model.save(
        dist_l2_mnist_config.model_dir, dist_l2_mnist_config.model_name
    )
