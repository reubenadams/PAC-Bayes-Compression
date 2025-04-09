import os
import wandb
from copy import deepcopy
import torch

from models import MLP
from distillation_comb import get_base_config, get_pacb_config
from config import FinalCompResults


def main():

    quick_test = False
    device = "cpu"
    dataset_name = "MNIST1D"
    seed = 0

    torch.manual_seed(seed)
    os.environ["WANDB_SILENT"] = "true"
    run = wandb.init()

    base_config = get_base_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        device=device
    )
    pacb_config = get_pacb_config(quick_test=quick_test)

    run.name = base_config.run_name
    run.save()

    print("Loading init and base models...")
    init_model = MLP(
        dimensions=base_config.model_dims,
        activation=base_config.hyperparams.activation,
        dropout_prob=base_config.hyperparams.dropout_prob,
        device=base_config.data.device,
    )
    base_model = deepcopy(init_model)
    init_model.load(base_config.model_init_dir, base_config.model_name)
    base_model.load(base_config.model_base_dir, base_config.model_name)

    # TODO: Use `compress_difference` and `threshold_codeword_length_to_reduce_k_means_max_iter` from a CompConfig
    print(f"{base_model.num_weights=}")
    final_quant_only_results = FinalCompResults()
    for codeword_length in range(1, 33):
        print(f"{codeword_length=}")
        quant_results = base_model.get_comp_pacb_results(
            delta=pacb_config.delta,
            train_loader=base_config.data.train_loader,
            test_loader=base_config.data.test_loader,
            C_domain=base_config.data.C_train_domain,
            C_data=base_config.data.C_train_data,
            codeword_length=codeword_length,
            compress_difference=True,
            init_model=init_model,
        )
        print(quant_results)
        final_quant_only_results.add_result(quant_results)
        quant_results.log()
    final_quant_only_results.save_to_json(filename=base_config.quant_metrics_path)

    # TODO: Get quant *and* low rank results.

    run.finish()


if __name__ == "__main__":
    main()
