import os


import torch
import wandb
from itertools import product
import cProfile
import pstats


from config import TrainConfig, DistTrainConfig, ExperimentConfig
from models import MLP
from load_data import get_dataloaders


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
torch.manual_seed(0)
os.environ["WANDB_SILENT"] = "true"

train_bases, test_dist_variance, train_dists = False, False, True

dataset_name = "MNIST1D"
# base_dims = [(40, d, 10) for d in [100, 200, 300, 400]]
base_dims = [(40, 500, 10)]
base_batch_sizes = [32, 64, 128]
base_lrs = [0.01, 0.0032, 0.001]

# base_dims = [(40, 100, 10)]
# base_batch_sizes = [32]
# base_lrs = [0.01]

dist_hidden_dims = [60, 61, 62, 63, 64, 66, 67, 68, 69, 70]
num_dist_attempts = 1

train_size, test_size = None, None


base_train_configs = {}
base_experiment_configs = {}
dist_experiment_configs = {}

for dims, batch_size, lr in product(base_dims, base_batch_sizes, base_lrs):
    base_train_configs[(batch_size, lr)] = TrainConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=2000,
        use_early_stopping=True,
        get_overall_train_loss=True,
        get_test_accuracy=True,
        train_loss_name="Base Train Loss",
        test_loss_name="Base Test Loss",
        test_accuracy_name="Base Test Accuracy",
    )
    base_experiment_configs[(dims, batch_size, lr)] = ExperimentConfig(
        project_name="Distillation MNIST1D Base",
        experiment="distillation",
        model_type="base",
        model_dims=dims,
        lr=lr,
        batch_size=batch_size,
        dataset_name="MNIST1D",
    )
    dist_experiment_configs[(dims, batch_size, lr)] = ExperimentConfig(
        project_name="Distillation MNIST1D Dist, Binary Search",
        experiment="distillation",
        model_type="dist",
        model_dims=dims,
        lr=lr,
        batch_size=batch_size,
        dataset_name="MNIST1D",
    )

dist_train_config = DistTrainConfig(use_whole_dataset=True, use_early_stopping=True)


def train_base_models():
    for dims, batch_size, lr in product(base_dims, base_batch_sizes, base_lrs):

        train_config = base_train_configs[(batch_size, lr)]
        experiment_config = base_experiment_configs[(dims, batch_size, lr)]

        if os.path.exists(experiment_config.model_path):
            print(f"File {experiment_config.model_path} found. Skipping model...")
            continue

        if train_config.log_with_wandb:
            wandb.init(
                project=experiment_config.project_name,
                name=f"{dims[1]}_{batch_size}_{lr}",
            )
        print(dims, batch_size, lr)
        torch.manual_seed(0)
        model = MLP(
            experiment_config.model_dims, experiment_config.model_act, device=device
        )
        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            experiment_config.dataset_name,
            experiment_config.batch_size,
            train_size=train_size,
            test_size=test_size,
        )
        overall_train_loss, target_loss_achieved = model.train(
            train_config,
            train_loader=train_loader,
            test_loader=test_loader,
            train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
            test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
            overall_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        )
        if train_config.log_with_wandb:
            wandb.finish()
        if target_loss_achieved:  # Only save if model reached target train loss
            print(
                f"Model reached target train loss {overall_train_loss} <= {train_config.target_overall_train_loss}"
            )
            model.save(experiment_config.model_dir, experiment_config.model_name)
        else:
            print(
                f"Model did not reach target train loss {overall_train_loss} > {train_config.target_overall_train_loss}"
            )


def dist_variance_test():

    for dims, batch_size, lr in product(base_dims, base_batch_sizes, base_lrs):
        base_experiment_config = base_experiment_configs[(dims, batch_size, lr)]
        dist_experiment_config = dist_experiment_configs[(dims, batch_size, lr)]

        for hidden_dim in dist_hidden_dims:

            if dist_train_config.log_with_wandb:
                wandb.finish()
                wandb.init(
                    project=dist_experiment_config.project_name,
                    name=f"{dims[1]}_{batch_size}_{lr}_{hidden_dim}",
                )

            print(
                f"Dims: {dims[1]}, Batch Size: {batch_size}, LR: {lr}, Dist Hidden Dim: {hidden_dim}"
            )

            base_model = MLP(
                base_experiment_config.model_dims,
                base_experiment_config.model_act,
                device=device,
            )

            try:
                base_model.load(base_experiment_config.model_path)
                print(
                    f"File {base_experiment_config.model_path} found. Loading model..."
                )
            except FileNotFoundError:
                print(
                    f"File {base_experiment_config.model_path} not found. Skipping model..."
                )
                continue

            torch.manual_seed(0)
            train_loader, test_loader = get_dataloaders(
                dataset_name=base_experiment_config.dataset_name,
                batch_size=dist_train_config.batch_size,
                train_size=train_size,
                test_size=test_size,
                use_whole_dataset=dist_train_config.use_whole_dataset,
                device=device,
            )

            kl_losses_and_epochs = base_model.get_dist_variance(
                dist_config=dist_train_config,
                domain_train_loader=train_loader,
                hidden_dim=hidden_dim,
                num_repeats=num_dist_attempts,
            )

            if dist_train_config.log_with_wandb:
                for trial, (kl_loss, num_epochs) in enumerate(kl_losses_and_epochs):
                    wandb.log(
                        {"Trial": trial, "KL Loss": kl_loss, "Num Epochs": num_epochs}
                    )
            print()


def train_dist_models():

    for dims, batch_size, lr in product(base_dims, base_batch_sizes, base_lrs):
        base_experiment_config = base_experiment_configs[(dims, batch_size, lr)]
        dist_experiment_config = dist_experiment_configs[(dims, batch_size, lr)]

        if dist_train_config.log_with_wandb:
            wandb.finish()
            wandb.init(
                project=dist_experiment_config.project_name,
                name=f"{dims[1]}_{batch_size}_{lr}",
                reinit=True,
            )

        print(dims, batch_size, lr)
        model_log = {
            "Dim": dims[1],
            "Batch Size": batch_size,
            "Learning Rate": lr,
        }
        base_model = MLP(
            base_experiment_config.model_dims,
            base_experiment_config.model_act,
            device=device,
        )

        try:
            base_model.load(base_experiment_config.model_path)
            print(f"File {base_experiment_config.model_path} found. Loading model...")
        except FileNotFoundError:
            print(
                f"File {base_experiment_config.model_path} not found. Skipping model..."
            )
            continue

        torch.manual_seed(0)
        train_loader, test_loader = get_dataloaders(
            dataset_name=base_experiment_config.dataset_name,
            batch_size=dist_train_config.batch_size,
            train_size=train_size,
            test_size=test_size,
            use_whole_dataset=dist_train_config.use_whole_dataset,
            device=device,
        )

        generalization_gap = base_model.get_generalization_gap(
            train_loader, test_loader
        )
        model_log["Generalization Gap"] = generalization_gap

        complexity = base_model.get_dist_complexity(
            dist_train_config,
            domain_train_loader=train_loader,
            num_attempts=num_dist_attempts,
        )
        model_log["Complexity"] = complexity

        # if complexity:
        #     print(
        #         f"Successfully distilled model. Complexity: {complexity}, Generalization Gap: {generalization_gap}"
        #     )

        if dist_train_config.log_with_wandb:
            wandb.log(model_log)

        print()
    if dist_train_config.log_with_wandb:
        wandb.finish()


if __name__ == "__main__":

    with cProfile.Profile() as pr:
        if train_bases:
            print("Training base")
            train_base_models()
        if test_dist_variance:
            print("Dist variance")
            dist_variance_test()
        if train_dists:
            print("Training dist")
            train_dist_models()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(10)
