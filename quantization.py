import os
import wandb
import torch
import pandas as pd

from distillation_comb import get_base_config, get_pacb_config, train_base_model
import config


def get_comp_config(quick_test: bool):
    return config.CompConfig.create(quick_test)


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
    comp_config = get_comp_config(quick_test=quick_test)

    run.name = base_config.run_name
    run.save()

    print("Training base model...")
    init_model, base_model, base_metrics = train_base_model(base_config=base_config)
    init_model.save(base_config.model_init_dir, base_config.model_name)

    if base_metrics.reached_target:  # Only save if model reached target train loss
        print("Model reached target train loss")
        base_model.save(base_config.model_base_dir, base_config.model_name)
    else:
        # If the model did not reach the target train loss log the metrics and finish the run early
        print("Model failed to reach target train loss")
        log_and_save_metrics(
            run_id=run.id,
            base_config=base_config,
            base_metrics=base_metrics,
        )
        run.finish()
        return

    # print("Loading init and base models...")
    # init_model = MLP(
    #     dimensions=base_config.model_dims,
    #     activation_name=base_config.hyperparams.activation_name,
    #     dropout_prob=base_config.hyperparams.dropout_prob,
    #     device=base_config.data.device,
    # )
    # base_model = deepcopy(init_model)
    # init_model.load(base_config.model_init_dir, base_config.model_name)
    # base_model.load(base_config.model_base_dir, base_config.model_name)

    # Get quant only results
    print("Getting quant only results...")
    if comp_config.get_quant_only_results:
        final_quant_only_results = config.FinalCompResults()
        print(f"Sensible codeword lengths: {base_model.sensible_codeword_lengths}")
        for codeword_length in range(1, comp_config.max_codeword_length + 1):  # TODO: Also use sensible codeword lengths so as not to use more clusters than weights?
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

    # Get low rank only results
    print("Getting low rank only results...")
    if comp_config.get_low_rank_only_results:
        final_low_rank_only_results = config.FinalCompResults()
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

    # Get low rank and quant results
    print("Getting low rank and quant results...")
    if comp_config.get_low_rank_and_quant_results:
        final_low_rank_and_quant_results = config.FinalCompResults()
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
    run.finish()

if __name__ == "__main__":
    main()
