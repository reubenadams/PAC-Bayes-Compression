import os
import json
import wandb
import torch
import pandas as pd

from distillation_comb import get_pacb_config, train_base_model
import config


def get_base_config(quick_test: bool, dataset_name: str, device: str, experiment_type: str) -> config.BaseConfig:
    hyperparams = config.BaseHyperparamsConfig.from_wandb_config(wandb.config)
    
    data_config = config.BaseDataConfig(dataset_name=dataset_name, device=device)
    # data_config.add_sample_sizes(quick_test)
    data_config.train_size = 1000 if quick_test else 50000
    data_config.test_size = 100 if quick_test else 10000
    data_config.add_dataloaders(hyperparams.batch_size)
    
    stopping_config = config.BaseStoppingConfig.create(quick_test)
    stopping_config.target_full_train_loss = None  # Base model should train until convergence
    stopping_config.patience = 1
    if quick_test:
        stopping_config.max_epochs = 1
    records = config.BaseRecordsConfig()
    return config.BaseConfig(
        hyperparams=hyperparams,
        data=data_config,
        stopping=stopping_config,
        records=records,
        experiment_type=experiment_type,
    )


def get_comp_config(quick_test: bool, base_config: config.BaseConfig) -> config.CompConfig:
    return config.CompConfig.create(
        quick_test=quick_test,
        dataset_name=base_config.data.dataset_name,
        device=base_config.data.device,
        new_input_shape=base_config.data._new_input_shape,
    )


def log_and_save_metrics(
        run_id: str,
        base_config: config.BaseConfig,
        pacb_config: config.PACBConfig = None,
        comp_config: config.CompConfig = None,
        base_metrics: config.BaseResults = None,
        pacb_metrics: config.PACBResults = None,
        final_low_rank_only_results: config.FinalCompResults = None,
        final_quant_only_results: config.FinalCompResults = None,
        final_low_rank_and_quant_results: config.FinalCompResults = None,
    ):
    all_configs = {"Run ID": run_id, "Run Name": base_config.run_name} | base_config.to_dict()
    if pacb_config is not None:
        all_configs |= pacb_config.to_dict()
    if comp_config is not None:
        all_configs |= comp_config.to_dict()

    base_metrics.log()
    all_metrics = base_metrics.to_dict()
    if pacb_metrics is not None:
        pacb_metrics.log()
        all_metrics |= pacb_metrics.to_dict()
    if final_low_rank_only_results is not None:
        final_low_rank_only_results.log()
        all_metrics |= final_low_rank_only_results.to_dict()
    if final_quant_only_results is not None:
        final_quant_only_results.log()
        all_metrics |= final_quant_only_results.to_dict()
    if final_low_rank_and_quant_results is not None:
        final_low_rank_and_quant_results.log()
        all_metrics |= final_low_rank_and_quant_results.to_dict()

    df = pd.DataFrame([all_configs | all_metrics])
    df.to_csv(base_config.quant_metrics_path, index=False)


def main():

    quick_test = False
    device = "cpu"
    dataset_name = "MNIST1D"
    seed = 0
    best_results = dict()

    torch.manual_seed(seed)
    os.environ["WANDB_SILENT"] = "true"
    run = wandb.init()

    base_config = get_base_config(
        quick_test=quick_test,
        dataset_name=dataset_name,
        device=device,
        experiment_type="quantization",
    )
    pacb_config = get_pacb_config(quick_test=quick_test)
    comp_config = get_comp_config(quick_test=quick_test, base_config=base_config)

    run.name = base_config.run_name
    run.save()

    print("Training base model...")
    init_model, base_model, base_metrics = train_base_model(base_config=base_config)
    init_model.save(base_config.model_init_dir, base_config.model_name)
    base_model.save(base_config.model_base_dir, base_config.model_name)
    comp_config.add_dataloaders(
        train_dataset=base_config.data.train_loader.dataset,
        test_dataset=base_config.data.test_loader.dataset,
        data_dir=base_config.data.data_dir,
    )
    comp_config.add_base_logit_loaders(
        base_model=base_model,
        train_dataset=base_config.data.train_loader.dataset,
        test_dataset=base_config.data.test_loader.dataset,
    )

    with torch.no_grad():

        # Get results without any compression        
        print()
        print("Getting results without any compression...")

        final_no_comp_results = config.FinalCompResults()
        num_union_bounds = 1

        no_comp_results = base_model.get_comp_pacb_results(
            delta=pacb_config.delta,
            num_union_bounds=num_union_bounds,
            train_loader=comp_config.train_loader,
            test_loader=comp_config.test_loader,
            rand_domain_loader=comp_config.rand_domain_loader,
            base_logit_train_loader=comp_config.base_logit_train_loader,
            base_logit_test_loader=comp_config.base_logit_test_loader,
            base_logit_rand_domain_loader=comp_config.base_logit_rand_domain_loader,
            C_domain=base_config.data.C_train_domain,
            C_data=base_config.data.C_train_data,
            ranks=None,
            codeword_length=None,
            compress_model_difference=False,
            init_model=None,
        )
        print(f"Bound inverse kl domain: {no_comp_results.error_bound_inverse_kl_spectral_domain}")
        print(f"Bound inverse kl data: {no_comp_results.error_bound_inverse_kl_spectral_data}")
        print(f"Bound pinsker domain: {no_comp_results.error_bound_pinsker_spectral_domain}")
        print(f"Bound pinsker data: {no_comp_results.error_bound_pinsker_spectral_data}")
        final_no_comp_results.add_result(no_comp_results)
        no_comp_results.log()
        final_no_comp_results.save_to_json(filename=base_config.no_comp_metrics_path)
        best_results["no_comp"] = no_comp_results.to_dict()

        # Get quant only results
        if comp_config.get_quant_only_results:
            
            print()
            print("Getting quant only results...")

            final_quant_only_results = config.FinalCompResults()
            codeword_lengths = [length for length in range(1, comp_config.max_codeword_length + 1) if length in base_model.get_sensible_codeword_lengths_quant_only()]
            num_union_bounds = len(codeword_lengths)
            print()
            print(f"{codeword_lengths=}")

            for codeword_length in codeword_lengths:
                print(f"{codeword_length=}")
                quant_results = base_model.get_comp_pacb_results(
                    delta=pacb_config.delta,
                    num_union_bounds=num_union_bounds,
                    train_loader=comp_config.train_loader,
                    test_loader=comp_config.test_loader,
                    rand_domain_loader=comp_config.rand_domain_loader,
                    base_logit_train_loader=comp_config.base_logit_train_loader,
                    base_logit_test_loader=comp_config.base_logit_test_loader,
                    base_logit_rand_domain_loader=comp_config.base_logit_rand_domain_loader,
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
            best_results["quant_only"] = final_quant_only_results.best_results.to_dict()

        # Get low rank only results
        if comp_config.get_low_rank_only_results:

            print()
            print("Getting low rank only results...")

            final_low_rank_only_results = config.FinalCompResults()
            rank_combs = base_model.get_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
            num_union_bounds = len(rank_combs)
            print(f"{rank_combs=}")

            for ranks in base_model.get_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step):
                print()
                print(f"{ranks=}")
                low_rank_results = base_model.get_comp_pacb_results(
                    delta=pacb_config.delta,
                    num_union_bounds=num_union_bounds,
                    train_loader=comp_config.train_loader,
                    test_loader=comp_config.test_loader,
                    rand_domain_loader=comp_config.rand_domain_loader,
                    base_logit_train_loader=comp_config.base_logit_train_loader,
                    base_logit_test_loader=comp_config.base_logit_test_loader,
                    base_logit_rand_domain_loader=comp_config.base_logit_rand_domain_loader,
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
            best_results["low_rank_only"] = final_low_rank_only_results.best_results.to_dict()

        # Get low rank and quant results
        if comp_config.get_low_rank_and_quant_results:

            print()
            print("Getting low rank and quant results...")

            final_low_rank_and_quant_results = config.FinalCompResults()
            codeword_lengths = base_model.get_sensible_codeword_lengths_low_rank_and_quant(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
            num_union_bounds = 0
            for ranks, code_lens in codeword_lengths.items():
                codeword_lengths[ranks] = [length for length in code_lens if length <= comp_config.max_codeword_length]
                num_union_bounds += len(codeword_lengths[ranks])
            print(f"{codeword_lengths=}")

            base_model.get_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
            for ranks in base_model.get_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step):
                for codeword_length in codeword_lengths[ranks]:
                    print()
                    print(f"{ranks=}")
                    print(f"\t{codeword_length=}")
                    low_rank_and_quant_results = base_model.get_comp_pacb_results(
                        delta=pacb_config.delta,
                        num_union_bounds=num_union_bounds,
                        train_loader=comp_config.train_loader,
                        test_loader=comp_config.test_loader,
                        rand_domain_loader=comp_config.rand_domain_loader,
                        base_logit_train_loader=comp_config.base_logit_train_loader,
                        base_logit_test_loader=comp_config.base_logit_test_loader,
                        base_logit_rand_domain_loader=comp_config.base_logit_rand_domain_loader,
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
            best_results["low_rank_and_quant"] = final_low_rank_and_quant_results.best_results.to_dict()

    log_and_save_metrics(
        run_id=run.id,
        base_config=base_config,
        pacb_config=pacb_config,
        comp_config=comp_config,
        base_metrics=base_metrics,
        final_low_rank_only_results=final_low_rank_only_results,
        final_quant_only_results=final_quant_only_results,
        final_low_rank_and_quant_results=final_low_rank_and_quant_results,
    )

    with open(base_config.best_comp_metrics_path, "w") as f:
        json.dump(best_results, f, indent=2)
    
    run.finish()

if __name__ == "__main__":
    main()
