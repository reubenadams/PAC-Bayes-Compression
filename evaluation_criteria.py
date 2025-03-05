import torch
from scipy import stats
from yaml import safe_load
import wandb


# Keep this
def get_hyp_vals(sweep_config_path: str) -> dict:
    hyp_vals = dict()
    with open(sweep_config_path, "r") as file:
        sweep_config = safe_load(file)
        for name, vals in sweep_config["parameters"].items():
            vals = vals["values"]
            if name not in ["input_dim", "output_dim"]:
                hyp_vals[name] = vals
    return hyp_vals


# Change this
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


def entropy(pmf: torch.Tensor) -> float:
    pmf_on_support = pmf[pmf != 0]
    return -torch.sum(pmf_on_support * torch.log(pmf_on_support))


def conditional_entropy(pmf_joint: torch.Tensor) -> float:
    """Takes a joint pmf p(x, y) and returns the conditional entropy H(x|y)"""
    pmf_y = pmf_joint.sum(0)
    pmf_y_expanded = pmf_joint.sum(0, keepdim=True).expand_as(pmf_joint)
    pmf_x_given_y = pmf_joint / pmf_y_expanded
    conditional_sums = torch.sum(pmf_x_given_y, dim=0)
    assert torch.allclose(conditional_sums, torch.ones_like(conditional_sums))
    entropies = [pmf_y[j] * entropy(pmf_x_given_y[:, j]) for j in range(pmf_joint.shape[1])]
    return torch.sum(torch.tensor(entropies))


# def mutual_inf(pmf1: torch.Tensor, pmf2: torch.Tensor, pmf_joint: torch.Tensor) -> float:
#     assert len(pmf1.shape) == 1
#     assert len(pmf2.shape) == 1
#     assert pmf_joint.shape == (pmf1.shape[0], pmf2.shape[0])
    
#     d1, d2 = pmf_joint.shape
#     pmf1 = pmf1.reshape(d1, 1).tile(1, d2).flatten()
#     pmf2 = pmf2.reshape(1, d2).tile(d1, 1).flatten()
#     pmf_joint = pmf_joint.flatten()

#     joint_support = pmf_joint != 0
#     pmf1 = pmf1[joint_support]
#     pmf2 = pmf2[joint_support]
#     pmf_joint = pmf_joint[joint_support]

#     return torch.sum(pmf_joint * torch.log(pmf_joint / (pmf1 * pmf2)))


def mutual_inf(pmf_joint: torch.Tensor) -> float:
    pmf0 = pmf_joint.sum(dim=1, keepdim=True).expand_as(pmf_joint).flatten()
    pmf1 = pmf_joint.sum(dim=0, keepdim=True).expand_as(pmf_joint).flatten()
    pmf_joint = pmf_joint.flatten()

    joint_support = pmf_joint != 0
    pmf0 = pmf0[joint_support]
    pmf1 = pmf1[joint_support]
    pmf_joint = pmf_joint[joint_support]

    return torch.sum(pmf_joint * torch.log(pmf_joint / (pmf0 * pmf1)))


def conditional_mutual_inf(pmf_joint_triple: torch.Tensor) -> float:
    """Takes a joint pmf p(x, y, z) and returns the conditional mutual information I(x;y|z)"""
    pmf_z = pmf_joint_triple.sum(0).sum(0)
    pmf_z_expanded = pmf_joint_triple.sum(0, keepdim=True).sum(1, keepdim=True).expand_as(pmf_joint_triple)
    pmf_xy_given_z = pmf_joint_triple / pmf_z_expanded
    conditional_sums = pmf_xy_given_z.sum(0).sum(0)
    assert torch.allclose(conditional_sums, torch.ones_like(conditional_sums))
    mut_infs = [pmf_z[k] * mutual_inf(pmf_xy_given_z[:, :, k]) for k in range(pmf_joint_triple.shape[2])]
    return torch.sum(torch.tensor(mut_infs))


if __name__ == "__main__":

    api = wandb.Api()
    sweep = api.sweep("teamreuben/2187-big/7spkiovz")
    runs = sweep.runs
    for run in runs:
        print(f"Run ID: {run.id}, Name: {run.name}")
        print(f"Config: {run.config}")
        print(f"Metrics: {run.summary}")
        break
    # hyp_vals = get_hyp_vals("sweep_config_distillation_base.yaml")
    # print(hyp_vals)
    # successes, complexities, gen_gaps = get_sweep_results(hyp_vals, "e8qaxkrj", "2187-big")
    # print(f"{successes.shape=}, {complexities.shape=}, {gen_gaps.shape=}")
