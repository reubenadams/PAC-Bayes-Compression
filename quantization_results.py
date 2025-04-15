import yaml
import json
import itertools

from config import BaseHyperparamsConfig


def main():
    with open('sweep_config_quant.yaml', 'r') as file:
        config = yaml.safe_load(file)

    parameters = config['parameters']
    param_names = list(parameters.keys())
    param_values = [parameters[name]['values'] for name in param_names]
    combinations = list(itertools.product(*param_values))

    hyp_configs = []
    for combo in combinations:
        param_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
        hyp_configs.append(BaseHyperparamsConfig(**param_dict))

    for hyp_config in hyp_configs:
        model_root_dir = "quantization/models/MNIST1D/40"
        run_name = hyp_config.run_name
        no_comp_metrics_path = f"{model_root_dir}/no_comp_metrics/{run_name}.json"
        low_rank_only_metrics_path = f"{model_root_dir}/low_rank_only_metrics/{run_name}.json"
        quant_only_metrics_path = f"{model_root_dir}/quant_only_metrics/{run_name}.json"
        low_rank_and_quant_metrics_path = f"{model_root_dir}/low_rank_and_quant_metrics/{run_name}.json"

        with open(no_comp_metrics_path, 'r') as file:
            no_comp_metrics = json.load(file)
            print(no_comp_metrics)
            assert False


if __name__ == "__main__":
    main()
