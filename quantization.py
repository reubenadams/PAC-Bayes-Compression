from distillation_end_to_end import *


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
    base_model = copy.deepcopy(init_model)
    init_model.load(base_config.model_init_dir, base_config.model_name)
    base_model.load(base_config.model_base_dir, base_config.model_name)

    for codeword_length in range(1, 33):
        quant_results = base_model.get_quantized_pacb_results(
            delta=pacb_config.delta,
            train_loader=base_config.data.train_loader,
            test_loader=base_config.data.test_loader,
            codeword_length=codeword_length,
        )
        quant_results.log()

    run.finish()


if __name__ == "__main__":
    main()
