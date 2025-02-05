import torch
import wandb

from models import MLP
from config import Config
from load_data import get_dataloaders


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


full_model_dims = [[784, d, 10] for d in [100, 200, 300]]
full_model_configs = [
    Config(experiment="distillation", model_type="full", model_dims=dims)
    for dims in full_model_dims
]

full_models = [
    MLP(config.model_dims, config.model_act, device=device)
    for config in full_model_configs
]

train_loader, test_loader = get_dataloaders(
    full_model_configs[0].dataset,
    full_model_configs[0].batch_size,
    train_size=100,
    test_size=100,
    new_size=full_model_configs[0].new_data_shape,
)


for full_config, full_model in zip(full_model_configs, full_models):

    try:

        full_model.load(full_config.model_path)
        print(f"File {full_config.model_path} found. Loading model...")

    except FileNotFoundError:
        wandb.init(project="Distillation complexity", name=f"{full_model.dimensions}")
        print(f"File {full_config.model_path} not found. Training model...")
        full_model.train(
            train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
            test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            lr=full_config.lr,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=full_config.epochs,
            get_overall_train_loss=True,
            overall_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            get_test_accuracy=True,
            train_loss_name="Full Train Loss",
            test_loss_name="Full Test Loss",
            test_accuracy_name="Full Test Accuracy",
            target_overall_train_loss=0.01,
        )
        full_model.save(full_config.model_dir, full_config.model_name)


dist_complexities = {}
for full_model in full_models:
    dist_complexity = full_model.get_dist_complexity(
        dim_skip=10,
        max_hidden_dim=100,
        dist_activation="relu",
        shift_logits=False,
        domain_train_loader=train_loader,
        lr=0.01,
        num_epochs=100,
        target_kl_on_train=0.01,
    )
    dist_complexities[full_model.dimensions[1]] = dist_complexity
    print(dist_complexity)
    print(f"Full Model Dim: {full_model.dimensions[1]}")
    print(f"Complexity: {dist_complexity}")

wandb.init(project="Distillation complexity", name="Complexity")
for dim, comp in dist_complexities.items():
    wandb.log({"Full Model Dim": dim, "Complexity": comp})
wandb.finish()

assert False

# Train the distillations
full_model.get_dists(
    dim_skip=10,
    dist_activation="relu",
    shift_logits=False,
    domain_train_loader=train_loader,
    domain_test_loader=train_loader,  # Note you've put the train loader here, since you want to know whether the NN learned a simple function *on the train set*
    data_test_loader=test_loader,
    lr=0.01,
    num_epochs=100,
    get_kl_on_test_data=True,
    get_accuracy_on_test_data=True,
)
