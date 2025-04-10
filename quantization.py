import os
import wandb
from copy import deepcopy
import torch

from models import MLP
from distillation_comb import get_base_config, get_pacb_config
from config import CompConfig, FinalCompResults


def get_comp_config(quick_test: bool):
    return CompConfig.create(quick_test)


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
        device=device
    )
    pacb_config = get_pacb_config(quick_test=quick_test)
    comp_config = get_comp_config(quick_test=quick_test)

    run.name = base_config.run_name
    run.save()

    print("Loading init and base models...")
    init_model = MLP(
        dimensions=base_config.model_dims,
        activation_name=base_config.hyperparams.activation_name,
        dropout_prob=base_config.hyperparams.dropout_prob,
        device=base_config.data.device,
    )
    base_model = deepcopy(init_model)
    init_model.load(base_config.model_init_dir, base_config.model_name)
    base_model.load(base_config.model_base_dir, base_config.model_name)

    # Get low rank only results
    if comp_config.get_low_rank_only_results:
        final_low_rank_only_results = FinalCompResults()
        base_model.populate_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
        for ranks in base_model.sensible_ranks:
            print(f"{ranks=}")
            low_rank_results = base_model.get_comp_pacb_results(
                delta=pacb_config.delta,
                train_loader=base_config.data.train_loader,
                test_loader=base_config.data.test_loader,
                C_domain=base_config.data.C_train_domain,
                C_data=base_config.data.C_train_data,
                ranks=ranks,
                compress_model_difference=comp_config.compress_model_difference,
                init_model=init_model,
            )
            final_low_rank_only_results.add_result(low_rank_results)
            low_rank_results.log()
        final_low_rank_only_results.get_best()
        final_low_rank_only_results.save_to_json(filename=base_config.low_rank_only_metrics_path)

    # Get quant only results
    if comp_config.get_quant_only_results:
        final_quant_only_results = FinalCompResults()
        print(f"Sensible codeword lengths: {base_model.sensible_codeword_lengths}")
        for codeword_length in range(1, comp_config.max_codeword_length + 1):  # TODO: Also use sensible codeword lengths so as not to use more clusters than weights?
            print(f"Codeword length: {codeword_length}")
            if codeword_length not in base_model.sensible_codeword_lengths:
                print(f"Skipping codeword length {codeword_length} as it is not sensible")
                continue
            print(f"{codeword_length=}")
            quant_results = base_model.get_comp_pacb_results(
                delta=pacb_config.delta,
                train_loader=base_config.data.train_loader,
                test_loader=base_config.data.test_loader,
                C_domain=base_config.data.C_train_domain,
                C_data=base_config.data.C_train_data,
                codeword_length=codeword_length,
                compress_model_difference=comp_config.compress_model_difference,
                init_model=init_model,
            )
            final_quant_only_results.add_result(quant_results)
            quant_results.log()
        final_quant_only_results.get_best()
        final_quant_only_results.save_to_json(filename=base_config.quant_only_metrics_path)

    # Get low rank and quant results
    if comp_config.get_low_rank_and_quant_results:
        final_low_rank_and_quant_results = FinalCompResults()
        base_model.populate_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
        for ranks in base_model.sensible_ranks:
            for codeword_length in range(1, comp_config.max_codeword_length + 1):
                if codeword_length not in base_model.sensible_codeword_lengths:
                    print(f"Skipping codeword length {codeword_length} as it is not sensible")
                    continue
                print(f"{ranks=}")
                print(f"\t{codeword_length=}")
                low_rank_and_quant_results = base_model.get_comp_pacb_results(
                    delta=pacb_config.delta,
                    train_loader=base_config.data.train_loader,
                    test_loader=base_config.data.test_loader,
                    C_domain=base_config.data.C_train_domain,
                    C_data=base_config.data.C_train_data,
                    ranks=ranks,
                    codeword_length=codeword_length,
                    compress_model_difference=comp_config.compress_model_difference,
                    init_model=init_model,
                )
                final_low_rank_and_quant_results.add_result(low_rank_and_quant_results)
                low_rank_and_quant_results.log()
        final_low_rank_and_quant_results.get_best()
        final_low_rank_and_quant_results.save_to_json(filename=base_config.low_rank_and_quant_metrics_path)

    run.finish()


if __name__ == "__main__":
    main()
