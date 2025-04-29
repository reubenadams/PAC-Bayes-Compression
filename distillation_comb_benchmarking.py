import os
import torch
import copy
import wandb
import io
import cProfile
import pstats

import config
from models import MLP


def get_base_config(quick_test: bool, dataset_name: str, device: str, experiment_type: str):
    hyperparams = config.BaseHyperparamsConfig(
        optimizer_name="adam",
        hidden_layer_width=512,
        num_hidden_layers=2,
        lr=0.003,
        batch_size=32,
        dropout_prob=0,
        weight_decay=0,
        activation_name="relu",
    )
    data_config = config.BaseDataConfig(dataset_name=dataset_name, device=device)
    data_config.add_sample_sizes(quick_test)
    data_config.add_dataloaders(hyperparams.batch_size)
    stopping_config = config.BaseStoppingConfig.create(quick_test)
    records = config.BaseRecordsConfig()
    return config.BaseConfig(
        hyperparams=hyperparams,
        data=data_config,
        stopping=stopping_config,
        records=records,
        experiment_type=experiment_type,
    )


def get_dist_config(
        quick_test: bool,
        dataset_name: str,
        use_whole_dataset: bool,
        device: str,
        base_config: config.BaseConfig,
        base_model: MLP
    ):
    hyperparams = config.DistHyperparamsConfig()
    data_config = config.DistDataConfig(
        dataset_name=dataset_name,
        use_whole_dataset=use_whole_dataset,
        device=device
    )
    data_config.add_sample_sizes(quick_test)
    data_config.add_dataloaders(
        new_input_shape=base_config.data.new_input_shape,
        train_dataset=base_config.data.train_loader.dataset,
        test_dataset=base_config.data.test_loader.dataset,
        data_filepath=base_config.data.data_filepath,
    )
    data_config.add_base_logit_loaders(
        base_model=base_model,
        train_dataset=base_config.data.train_loader.dataset,
        test_dataset=base_config.data.test_loader.dataset,
    )
    stopping_config = config.DistStoppingConfig.create(quick_test)
    objective = config.DistObjectiveConfig()
    records = config.DistRecordsConfig()
    return config.DistConfig(
        hyperparams=hyperparams,
        stopping=stopping_config,
        objective=objective,
        records=records,
        data=data_config
    )


def train_base_model(
        base_config: config.BaseConfig,
    ) -> tuple[MLP, MLP, config.BaseResults]:

    init_model = MLP(
        dimensions=base_config.model_dims,
        activation_name=base_config.hyperparams.activation_name,
        dropout_prob=base_config.hyperparams.dropout_prob,
        device=base_config.data.device,
    )
    base_model = copy.deepcopy(init_model)
    print("init_model: ")
    print(init_model)
    print("base_model: ")
    print(base_model)

    base_metrics = base_model.train_model(
        base_config=base_config,
        train_loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
        test_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        full_train_loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
    )

    return init_model, base_model, base_metrics


def train_dist_model(
        base_model: MLP,
        dist_config: config.DistConfig,
    ) -> tuple[MLP, config.DistFinalResults]:

    base_model.eval()
    dist_model, dist_metrics = base_model.get_dist_complexity(dist_config=dist_config)
    return dist_model, dist_metrics


def profile_train_dist_model(base_model, dist_config):
    # Create a profiler object
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    # Run the function to profile
    dist_model, dist_metrics = train_dist_model(base_model=base_model, dist_config=dist_config)
    
    # Stop profiling
    profiler.disable()
    
    # Create a StringIO object to hold the profiling results
    s = io.StringIO()
    
    # Sort the results by cumulative time
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    
    # Print the stats to our StringIO object
    ps.print_stats(30)  # Print top 30 time-consuming operations
    
    # Get the string from the StringIO object
    profile_result = s.getvalue()
    
    # Print the profiling results
    print("Profiling Results for train_dist_model:")
    print(profile_result)


def main():

    quick_test = True
    device = "cpu"
    dataset_name = "MNIST1D"
    seed = 0

    torch.manual_seed(seed)
    os.environ["WANDB_SILENT"] = "true"
    run = wandb.init()

    base_config = get_base_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        device=device,
        experiment_type="distillation",
    )

    print("Training base model...")
    init_model, base_model, base_metrics = train_base_model(base_config=base_config)

    dist_config = get_dist_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        use_whole_dataset=True,  # TODO: I think there should be a better place for this? Note it depends on the dataset.
        device=device,
        base_config=base_config,
        base_model=base_model,
    )

    print("Distilling model with profiling...")
    # Use the profiling wrapper instead of direct function call
    profile_train_dist_model(
            base_model=base_model,
            dist_config=dist_config,
        )

    run.finish()


if __name__ == "__main__":
    main()
