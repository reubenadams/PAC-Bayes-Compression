from distillation_end_to_end import *
from kl_utils import pacb_error_bound_inverse_kl, pacb_error_bound_pinsker


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
        quant_model = base_model.get_quantized_model(codeword_length=codeword_length)
        quant_train_accuracy = quant_model.get_full_accuracy(base_config.data.train_loader)
        quant_test_accuracy = quant_model.get_full_accuracy(base_config.data.test_loader)
        quant_KL = base_model.KL_of_quantized_model(codeword_length=codeword_length)
        quant_pacb_error_bound_inverse_kl = pacb_error_bound_inverse_kl(delta=pacb_config.delta, KL=quant_KL, n=base_config.data.train_size)
        quant_pacb_error_bound_pinsker = pacb_error_bound_pinsker(empirical_error=quant_train_accuracy, KL=quant_KL, n=base_config.data.train_size, delta=pacb_config.delta)
        perturbation = 




    run.finish()


if __name__ == "__main__":
    main()
