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
        stopping_config.max_epochs = 2
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

        final_no_comp_results: config.FinalCompResults = None,
        final_quant_k_means_results: config.FinalCompResults = None,
        final_quant_trunc_results: config.FinalCompResults = None,
        
        final_low_rank_results: config.FinalCompResults = None,
        final_low_rank_and_quant_k_means_results: config.FinalCompResults = None,
        final_low_rank_and_quant_trunc_results: config.FinalCompResults = None,
    ) -> None:
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

    if final_no_comp_results is not None:
        final_no_comp_results.log()
        all_metrics |= final_no_comp_results.to_dict()
    if final_quant_k_means_results is not None:
        final_quant_k_means_results.log()
        all_metrics |= final_quant_k_means_results.to_dict()
    if final_quant_trunc_results is not None:
        final_quant_trunc_results.log()
        all_metrics |= final_quant_trunc_results.to_dict()

    if final_low_rank_results is not None:
        final_low_rank_results.log()
        all_metrics |= final_low_rank_results.to_dict()
    if final_low_rank_and_quant_k_means_results is not None:
        final_low_rank_and_quant_k_means_results.log()
        all_metrics |= final_low_rank_and_quant_k_means_results.to_dict()
    if final_low_rank_and_quant_trunc_results is not None:
        final_low_rank_and_quant_trunc_results.log()
        all_metrics |= final_low_rank_and_quant_trunc_results.to_dict()

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
        data_filepath=base_config.data.data_filepath,
    )
    comp_config.add_base_logit_loaders(
        base_model=base_model,
        train_dataset=base_config.data.train_loader.dataset,
        test_dataset=base_config.data.test_loader.dataset,
    )

    with torch.no_grad():

        # Results without compression
        if comp_config.get_no_comp_results:

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
                exponent_bits=None,
                mantissa_bits=None,
                compress_model_difference=False,
                init_model=None,
            )
            final_no_comp_results.add_result(no_comp_results)
            no_comp_results.log()
            final_no_comp_results.get_best()
            final_no_comp_results.save_to_json(filename=base_config.no_comp_metrics_path)
            best_results["no_comp"] = final_no_comp_results.best_results.to_dict()


        # Quant k-means results
        if comp_config.get_quant_k_means_results:

            print()
            print("Getting quant k-means results...")

            final_quant_k_means_results = config.FinalCompResults()
            sensible_ranks_and_codeword_lengths = [length for length in range(1, comp_config.max_codeword_length + 1) if length in base_model.get_sensible_codeword_lengths()]
            num_union_bounds = len(sensible_ranks_and_codeword_lengths)
            print(f"{sensible_ranks_and_codeword_lengths=}")

            for codeword_length in sensible_ranks_and_codeword_lengths:
                print(f"\t{codeword_length=}")
                quant_k_means_results = base_model.get_comp_pacb_results(
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
                    codeword_length=codeword_length,
                    exponent_bits=None,
                    mantissa_bits=None,
                    compress_model_difference=comp_config.compress_model_difference,
                    init_model=init_model,
                )
                final_quant_k_means_results.add_result(quant_k_means_results)
                quant_k_means_results.log()
            final_quant_k_means_results.get_best()
            final_quant_k_means_results.save_to_json(filename=base_config.quant_k_means_metrics_path)
            best_results["quant_k_means"] = final_quant_k_means_results.best_results.to_dict()


        # Quant truncation results
        if comp_config.get_quant_trunc_results:

            print()
            print("Getting quant truncation results...")
            
            final_quant_trunc_results = config.FinalCompResults()
            num_union_bounds = 8 * 23  # 8 bits for exponent, 23 bits for mantissa

            # TODO: Surely we don't want to cover *all* values of b_e and b_m?
            for b_e in range(9):
                for b_m in range(24):
                    print(f"\t{b_e=}, {b_m=}")
                    quant_trunc_results = base_model.get_comp_pacb_results(
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
                        exponent_bits=b_e,
                        mantissa_bits=b_m,
                        compress_model_difference=comp_config.compress_model_difference,
                        init_model=init_model,
                    )
                    final_quant_trunc_results.add_result(quant_trunc_results)
                    quant_trunc_results.log()
            final_quant_trunc_results.get_best()
            final_quant_trunc_results.save_to_json(filename=base_config.quant_trunc_metrics_path)
            best_results["quant_trunc"] = final_quant_trunc_results.best_results.to_dict()


        # Low rank results
        if comp_config.get_low_rank_results:

            print()
            print("Getting low rank results...")

            final_low_rank_results = config.FinalCompResults()
            rank_combs = base_model.get_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
            num_union_bounds = len(rank_combs)
            print(f"{rank_combs=}")

            for ranks in rank_combs:
                print(f"\t{ranks=}")
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
                    codeword_length=None,
                    exponent_bits=None,
                    mantissa_bits=None,
                    compress_model_difference=comp_config.compress_model_difference,
                    init_model=init_model,
                )
                final_low_rank_results.add_result(low_rank_results)
                low_rank_results.log()
            final_low_rank_results.get_best()
            final_low_rank_results.save_to_json(filename=base_config.low_rank_metrics_path)
            best_results["low_rank"] = final_low_rank_results.best_results.to_dict()


        # Low rank and quant k-means results
        if comp_config.get_low_rank_and_quant_k_means_results:

            print()
            print("Getting low rank and quant k-means results...")

            final_low_rank_and_quant_k_means_results = config.FinalCompResults()
            sensible_ranks_and_codeword_lengths = base_model.get_sensible_ranks_and_codeword_lengths(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
            sensible_ranks_and_codeword_lengths = [(ranks, code_len) for ranks, code_len in sensible_ranks_and_codeword_lengths if code_len <= comp_config.max_codeword_length]
            num_union_bounds = len(sensible_ranks_and_codeword_lengths)
            print(f"{sensible_ranks_and_codeword_lengths=}")

            for ranks, codeword_length in sensible_ranks_and_codeword_lengths:
                print(f"\t{ranks=}, {codeword_length=}")
                low_rank_and_quant_k_means_results = base_model.get_comp_pacb_results(
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
                    exponent_bits=None,
                    mantissa_bits=None,
                    compress_model_difference=comp_config.compress_model_difference,
                    init_model=init_model,
                )
                final_low_rank_and_quant_k_means_results.add_result(low_rank_and_quant_k_means_results)
                low_rank_and_quant_k_means_results.log()
            final_low_rank_and_quant_k_means_results.get_best()
            final_low_rank_and_quant_k_means_results.save_to_json(filename=base_config.low_rank_and_quant_k_means_metrics_path)
            best_results["low_rank_and_quant_k_means"] = final_low_rank_and_quant_k_means_results.best_results.to_dict()


        # Low rank and quant truncation results
        if comp_config.get_low_rank_and_quant_k_means_results:

            print()
            print("Getting low rank and quant truncation results...")

            final_low_rank_and_quant_trunc_results = config.FinalCompResults()
            rank_combs = base_model.get_sensible_ranks(min_rank=comp_config.min_rank, rank_step=comp_config.rank_step)
            num_union_bounds = 8 * 23 * len(rank_combs)  # TODO: Should we really use all possibilities for b_e and b_m?

            for ranks in rank_combs:
                print(f"\t{ranks=}")
                for b_e in range(9):
                    for b_m in range(24):
                        print(f"\t\t{b_e=}, {b_m=}")
                        low_rank_and_quant_trunc_results = base_model.get_comp_pacb_results(
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
                            exponent_bits=None,
                            mantissa_bits=None,
                            compress_model_difference=comp_config.compress_model_difference,
                            init_model=init_model,
                        )
                        final_low_rank_and_quant_trunc_results.add_result(low_rank_and_quant_trunc_results)
                        low_rank_and_quant_trunc_results.log()
            final_low_rank_and_quant_trunc_results.get_best()
            final_low_rank_and_quant_trunc_results.save_to_json(filename=base_config.low_rank_and_quant_trunc_metrics_path)
            best_results["low_rank_and_quant_trunc"] = final_low_rank_and_quant_trunc_results.best_results.to_dict()


    log_and_save_metrics(
        run_id=run.id,
        base_config=base_config,
        pacb_config=pacb_config,
        comp_config=comp_config,
        base_metrics=base_metrics,

        final_no_comp_results=final_no_comp_results,
        final_quant_k_means_results=final_quant_k_means_results,
        final_quant_trunc_results=final_quant_trunc_results,

        final_low_rank_results=final_low_rank_results,
        final_low_rank_and_quant_k_means_results=final_low_rank_and_quant_k_means_results,
        final_low_rank_and_quant_trunc_results=final_low_rank_and_quant_trunc_results,
    )

    with open(base_config.best_comp_metrics_path, "w") as f:
        json.dump(best_results, f, indent=2)
    
    run.finish()

if __name__ == "__main__":
    main()
