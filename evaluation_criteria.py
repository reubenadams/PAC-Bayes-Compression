import torch
from scipy import stats
import yaml
import wandb


def get_hyp_vals(sweep_config_path: str) -> dict:
    hyp_vals = dict()
    with open(sweep_config_path, "r") as file:
        sweep_config = yaml.safe_load(file)
        for name, vals in sweep_config["parameters"].items():
            vals = vals["values"]
            if name not in ["input_dim", "output_dim"]:
                hyp_vals[name] = vals
    return hyp_vals


def get_sweep_results(hyp_vals: dict, dist_sweep_id: str, project_name: str, entity: str = "teamreuben") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    hyp_names = sorted(list(hyp_vals.keys()))  # TODO: Make sure we're always sorting.
    
    results_shape = tuple(len(hyp_vals[name]) for name in hyp_names)
    successes = torch.full(results_shape, False)  # To track which runs actually ran and actually met target loss
    complexities = torch.zeros(results_shape)
    gen_gaps = torch.zeros(results_shape)

    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project_name}/{dist_sweep_id}")
    runs = sweep.runs
    
    for run in runs:
        config = [(name, run.config[name]) for name in hyp_names]  # List to ensure order
        index = tuple(hyp_vals[name].index(val) for name, val in config)
        complexities[index] = run.summary["Complexity"]
        gen_gaps[index] = run.summary["Generalization Gap"]
        successes[index] = True  # N.B. Existance of run is sufficient to say model reached target loss otherwise base run would not have saved model.
    
    return successes, complexities, gen_gaps


def get_krcc(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> float:
    """Takes three 3x3x3x3x3x3x3 arrays (one dim for each hyperparameter), returns the
    Kendall rank correlation coefficient between complexities and generalization gaps where successes is True"""
    assert successes.shape == complexities.shape == gen_gaps.shape
    complexities = complexities[successes].flatten()
    gen_gaps = gen_gaps[successes].flatten()
    return stats.kendalltau(complexities, gen_gaps).statistic


def get_granulated_krccs(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> float:
    """Takes three 3x3x3x3x3x3x3 arrays (one dim for each hyperparameter), returns the *granulated*
    Kendall rank correlation coefficient*s* between complexities and generalization gaps where successes is True"""
    assert successes.shape == complexities.shape == gen_gaps.shape
    granulated_krccs = []
    for dim in range(len(successes.shape)):
        successes_flattened = flatten_except_dim(successes, dim)
        complexities_flattened = flatten_except_dim(complexities, dim)
        gen_gaps_flattened = flatten_except_dim(gen_gaps, dim)
        krccs_for_dim = []

        num_krccs = 0
        num_combs = successes_flattened.shape[0]
        total_krcc = 0
        for success_batch, comp_batch, gap_batch in zip(successes_flattened, complexities_flattened, gen_gaps_flattened):
            if not success_batch.all():
                print("Oh dear, no success to measure KRCC across.")
            else:
                num_krccs += 1
                comp_batch = comp_batch[success_batch]
                gap_batch = gap_batch[success_batch]
                krcc = get_krcc(success_batch, comp_batch, gap_batch)
                total_krcc += krcc
                krccs_for_dim.append(krcc)
        granulated_krccs.append(total_krcc / num_krccs)
        if num_krccs != num_combs:
            print(f"Oh dear, not all of the batches contributed to the granulated KRCC for dim {dim}")
    return granulated_krccs


def flatten_except_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tensor = tensor.moveaxis(dim, 0)
    tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor.transpose(0, 1)


if __name__ == "__main__":
    # base_hyp_vals = get_hyp_vals("sweep_config_distillation_base.yaml")
    # print(base_hyp_vals)
    # print()
    # dist_hyp_vals = get_hyp_vals("sweep_config_distillation_dist.yaml")
    # print(dist_hyp_vals)
    # print()
    # successes, complexities, gen_gaps = get_sweep_results(hyp_vals=dist_hyp_vals, dist_sweep_id="nrmfqbij", project_name="big-run")
    # print(successes)
    # print(complexities)
    # print(gen_gaps)

    a = torch.arange(24).reshape(2, 3, 4)
    print(a)
    print(flatten_except_dim(a, 0))
    print(flatten_except_dim(a, 1))
    print(flatten_except_dim(a, 2))