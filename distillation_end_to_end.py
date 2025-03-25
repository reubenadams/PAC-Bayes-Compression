import os
import torch
import wandb
import argparse
import copy

from config import BaseTrainConfig, DistTrainConfig, ExperimentConfig
from models import MLP
from load_data import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train and distill neural network models")
    parser.add_argument("--toy_run", action="store_true", help="Run with smaller dataset for testing")
    parser.add_argument("--device", type=str, default="cpu", help="Specify device (cuda/cpu)")
    parser.add_argument("--dataset", type=str, default="MNIST1D", help="Dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pickle", action="store_true", default=True, help="Pickle results locally (default: True)")
    return parser.parse_args()


def setup_environment(seed):
    torch.manual_seed(seed)
    os.environ["WANDB_SILENT"] = "true"


def get_base_run_params(toy_run):

    base_run_params = {
        "use_early_stopping": True,
        "get_full_train_loss": True,
        "get_final_train_loss": True,
        "get_final_test_loss": True,
        "get_final_train_accuracy": True,
        "get_final_test_accuracy": True,
        "train_loss_name": "Base Train Loss",
        "test_loss_name": "Base Test Loss",
        "train_accuracy_name": "Base Train Accuracy",
        "test_accuracy_name": "Base Test Accuracy",
        }

    if toy_run:
        base_run_params |= {
            "train_size": 100,
            "test_size": 100,
            "num_epochs": 200,
            "patience": 10,
            "target_full_train_loss": 0.1
        }
    else:
        base_run_params |= {
            "train_size": None,
            "test_size": None,
            "num_epochs": 1000000,
            "patience": 1000,
            "target_full_train_loss": 0.01
        }

    return base_run_params


def get_dist_run_params(toy_run):
    if toy_run:
        dist_run_params = {
            "train_size": 100,
            "test_size": 100,
            "max_epochs": 10000,
            "patience": 10,
            "target_kl_on_train": 0.1,
            "num_dist_attempts": 1
        }
    else:
        dist_run_params = {
            "train_size": None,
            "test_size": None,
            "max_epochs": 100000,
            "patience": 100,
            "target_kl_on_train": 0.01,
            "num_dist_attempts": 5
        }
    return dist_run_params


def get_hyperparams():
    hyperparams = {
        "optimizer": wandb.config.optimizer_name,
        "hidden_layer_width": wandb.config.hidden_layer_width,
        "num_hidden_layers": wandb.config.num_hidden_layers,
        "learning_rate": wandb.config.lr,
        "batch_size": wandb.config.batch_size,
        "dropout_prob": wandb.config.dropout_prob,
        "weight_decay": wandb.config.weight_decay,
    }
    return hyperparams


def get_run_name(hyperparams):
    run_name = (f"op{hyperparams['optimizer']}_"
                f"hw{hyperparams['hidden_layer_width']}_"
                f"nl{hyperparams['num_hidden_layers']}_"
                f"lr{hyperparams['learning_rate']}_"
                f"bs{hyperparams['batch_size']}_"
                f"dp{hyperparams['dropout_prob']}_"
                f"wd{hyperparams['weight_decay']}")
    return run_name


def get_base_train_config(hyperparams, base_run_params):
    base_train_config = BaseTrainConfig(
        optimizer_name=hyperparams["optimizer"],
        lr=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        dropout_prob=hyperparams["dropout_prob"],
        weight_decay=hyperparams["weight_decay"],
        num_epochs=base_run_params["num_epochs"],
        use_early_stopping=base_run_params["use_early_stopping"],
        target_full_train_loss=base_run_params["target_full_train_loss"],
        patience=base_run_params["patience"],
        get_full_train_loss=base_run_params["get_full_train_loss"],
        get_full_train_accuracy=base_run_params["get_train_accuracy"],
        get_full_test_accuracy=base_run_params["get_test_accuracy"],
        get_final_train_loss = True,
        get_final_test_loss = True,
        get_final_train_accuracy = True,
        get_final_test_accuracy = True,
        train_loss_name=base_run_params,
        test_loss_name=base_run_params,
        train_accuracy_name=base_run_params["train_accuracy_name"],
        test_accuracy_name=base_run_params["test_accuracy_name"],
        )
    return base_train_config


def get_dist_train_config(dist_run_params):
    dist_train_config = DistTrainConfig(
        max_epochs=dist_run_params["max_epochs"],
        use_whole_dataset=True,
        use_early_stopping=True,
        target_kl_on_train=dist_run_params["target_kl_on_train"],
        patience=dist_run_params["patience"],
    )
    return dist_train_config


def get_base_experiment_config(hyperparams, model_dims, dataset_name, model_name):
    base_experiment_config = ExperimentConfig(
        experiment="distillation",
        model_type="base",
        model_dims=model_dims,
        optimizer_name=hyperparams["optimizer"],
        lr=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        dropout_prob=hyperparams["dropout_prob"],
        weight_decay=hyperparams["weight_decay"],
        dataset_name=dataset_name,
        model_name=model_name,
    )
    return base_experiment_config


def get_dist_experiment_config(hyperparams, model_dims, dataset_name, model_name):
    dist_experiment_config = ExperimentConfig(
        experiment="distillation",
        model_type="dist",
        model_dims=model_dims,
        lr=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        dropout_prob=hyperparams["dropout_prob"],
        weight_decay=hyperparams["weight_decay"],
        dataset_name=dataset_name,
        model_name=model_name,
    )
    return dist_experiment_config


def initialize_wandb_and_configs(toy_run, dataset_name):
    """Initialize wandb and create configuration objects"""
    run = wandb.init()
    run_params = get_base_run_params(toy_run)
    hyperparams = get_hyperparams()
    run.name = get_run_name(hyperparams)
    run.save()

    model_dims = [wandb.config.input_dim] + [wandb.config.hidden_layer_width] * wandb.config.num_hidden_layers + [wandb.config.output_dim]
    model_name = wandb.run.name

    base_train_config = get_base_train_config(hyperparams, run_params)
    base_experiment_config = get_base_experiment_config(hyperparams, model_dims, dataset_name, model_name)

    return base_train_config, base_experiment_config


def train_base_model(
        base_experiment_config: ExperimentConfig,
        base_train_config: BaseTrainConfig,
        device,
        train_size,
        test_size
        ):

    torch.manual_seed(0)
    init_model = MLP(
        dimensions=base_experiment_config.model_dims,
        activation=base_experiment_config.model_act,
        dropout_prob=base_experiment_config.dropout_prob,
        device=device
    )
    init_model.save(base_experiment_config.model_init_dir, base_experiment_config.model_name)
    base_model = copy.deepcopy(init_model)

    torch.manual_seed(0)
    train_loader, test_loader = get_dataloaders(
        base_experiment_config.dataset_name,
        base_experiment_config.batch_size,
        train_size=train_size,
        test_size=test_size,
    )

    # full_train_loss, reached_target, lost_patience, epochs_taken = model.train_model(
    base_train_metrics = base_model.train_model(
        train_config=base_train_config,
        train_loader=train_loader,
        test_loader=test_loader,
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        full_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
    )

    if base_train_metrics.reached_target:  # Only save if model reached target train loss
        print("Model reached target train loss")
        base_model.save(base_experiment_config.model_trained_dir, base_experiment_config.model_name)
    else:
        print("Model failed to reach target train loss")

    return init_model, base_model, base_train_metrics
    # wandb.log({
    #     "Final Full " + base_train_config.train_loss_name: final_metrics.full_train_loss,
    #     "Reached Target": final_metrics.reached_target,
    #     "Lost Patience": final_metrics.lost_patience,
    #     "Ran out of epochs": not (final_metrics.reached_target or final_metrics.lost_patience),
    #     "Epochs Taken": final_metrics.epochs_taken,
    # })
    # wandb.finish()


def train_dist_model(
        base_model: MLP,
        base_experiment_config: ExperimentConfig,
        dist_experiment_config: ExperimentConfig,
        dist_train_config: DistTrainConfig,
        dist_run_params,
        device,
        train_size,
        test_size
        ):

    base_model.eval()

    # TODO: Decide whether we need to repeat this for some reason?
    torch.manual_seed(0)
    train_loader, test_loader = get_dataloaders(
        dataset_name=base_experiment_config.dataset_name,
        batch_size=dist_train_config.batch_size,
        train_size=train_size,
        test_size=test_size,
        use_whole_dataset=dist_train_config.use_whole_dataset,
        device=device,
    )

    complexity, dist_model = base_model.get_dist_complexity(
        dist_train_config,
        domain_train_loader=train_loader,
        num_attempts=dist_run_params["num_dist_attempts"],
    )
    # model_log["Complexity"] = complexity

    dist_model.save(dist_experiment_config.model_dir, dist_experiment_config.model_name)

    # wandb.log(model_log)
    # wandb.finish()


def record_everything():
    pass
    # hyperparams
    # run_params
    # dist_params
    # base_train_metrics
    # dist_train_metrics
    # pac bounds


def main():
    args = parse_args()
    setup_environment(args.seed)
    run_params = get_base_run_params(args.toy_run)
    device = torch.device(args.device)
    base_train_config, base_experiment_config = initialize_wandb_and_configs(run_params, args.dataset)
    os.makedirs("trained_models", exist_ok=True)

    print("Training base model...")
    train_base_model(
        base_train_config=base_train_config,
        base_experiment_config=base_experiment_config,
        device=device,
        train_size=run_params["train_size"],
        test_size=run_params["test_size"],
    )


if __name__ == "__main__":
    main()