import torch
from scipy import stats
from yaml import safe_load
import wandb
from itertools import product


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


def epsilon_oracle(gen_gaps: torch.Tensor, epsilon: float) -> torch.Tensor:
    return gen_gaps + torch.randn(gen_gaps.shape) * epsilon


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


def get_differences(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.ndim == 1
    num_elements = tensor.size(0)
    diffs =  torch.tensor([tensor[i] - tensor[j] for i, j in product(range(num_elements), repeat=2) if i != j])
    assert diffs.numel() == num_elements * (num_elements - 1)
    return diffs


def get_signs(tensor: torch.Tensor) -> torch.Tensor:
    diffs = get_differences(tensor)
    diffs[diffs == 0] = 1
    return torch.sign(diffs)


def get_joint_probs_from_signs(complexities_signs: torch.Tensor, gen_gaps_signs: torch.Tensor, prob_hyp1_hyp2: torch.Tensor) -> tuple[torch.Tensor]:
    prob_nn = ((complexities_signs == -1) & (gen_gaps_signs == -1)).float().mean() * prob_hyp1_hyp2
    prob_np = ((complexities_signs == -1) & (gen_gaps_signs == 1)).float().mean() * prob_hyp1_hyp2
    prob_pn = ((complexities_signs == 1) & (gen_gaps_signs == -1)).float().mean() * prob_hyp1_hyp2
    prob_pp = ((complexities_signs == 1) & (gen_gaps_signs == 1)).float().mean() * prob_hyp1_hyp2
    return prob_nn, prob_np, prob_pn, prob_pp


def get_joint_probs(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor, slices: tuple[slice], prob_hyp1_hyp2:torch.Tensor):
    successes_slice = successes[slices].flatten()
    complexities_slice = complexities[slices].flatten()
    gen_gaps_slice = gen_gaps[slices].flatten()

    complexities_slice = complexities_slice[successes_slice]
    gen_gaps_slice = gen_gaps_slice[successes_slice]
    assert complexities_slice.shape == gen_gaps_slice.shape

    complexities_signs = get_signs(complexities_slice)
    gen_gaps_signs = get_signs(gen_gaps_slice)
    assert complexities_signs.shape == gen_gaps_signs.shape

    return get_joint_probs_from_signs(complexities_signs, gen_gaps_signs, prob_hyp1_hyp2)


def get_pmf_joint_triple_Vmu_Vg_US(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor, hyp_dim1: int, hyp_dim2: int) -> torch.Tensor:
    
    assert successes.shape == complexities.shape == gen_gaps.shape
    num_vals_at_dim1 = successes.size(hyp_dim1)
    num_vals_at_dim2 = successes.size(hyp_dim2)
    assert num_vals_at_dim1 == num_vals_at_dim2 == 3
    
    pmf_joint_triple = torch.zeros((2, 2, num_vals_at_dim1 * num_vals_at_dim2))  # Vmu, Vg \in {-1, 1}, and 9 vals for the conditioned hyperparameters
    prob_hyp1_hyp2 = 1 / (num_vals_at_dim1 * num_vals_at_dim2)
    
    for idx1 in range(num_vals_at_dim1):
        for idx2 in range(num_vals_at_dim2):

            slices = [slice(None)] * successes.ndim
            slices[hyp_dim1] = idx1
            slices[hyp_dim2] = idx2

            prob_nn, prob_np, prob_pn, prob_pp = get_joint_probs(successes, complexities, gen_gaps, slices, prob_hyp1_hyp2)

            pmf_joint_triple[0, 0, num_vals_at_dim1 * idx1 + idx2] = prob_nn
            pmf_joint_triple[0, 1, num_vals_at_dim1 * idx1 + idx2] = prob_np
            pmf_joint_triple[1, 0, num_vals_at_dim1 * idx1 + idx2] = prob_pn
            pmf_joint_triple[1, 1, num_vals_at_dim1 * idx1 + idx2] = prob_pp
    
    return pmf_joint_triple


def get_normalized_conditional_entropies(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> torch.Tensor:
    num_hyp_dims = successes.ndim
    normed_cond_entropies = []
    for hyp_dim1 in range(num_hyp_dims):
        for hyp_dim2 in range(num_hyp_dims):
            if hyp_dim1 == hyp_dim2:
                continue
            pmf_joint_triple_Vmu_Vg_US = get_pmf_joint_triple_Vmu_Vg_US(successes, complexities, gen_gaps, hyp_dim1, hyp_dim2)
            pmf_joint_Vg_US = pmf_joint_triple_Vmu_Vg_US.sum(dim=0)
            cond_mut_inf = conditional_mutual_inf(pmf_joint_triple=pmf_joint_triple_Vmu_Vg_US)
            cond_entropy = conditional_entropy(pmf_joint=pmf_joint_Vg_US)
            normed_cond_entropies.append(cond_mut_inf / cond_entropy)
    assert len(normed_cond_entropies) == num_hyp_dims * (num_hyp_dims - 1) // 2
    return torch.tensor(normed_cond_entropies)


def CIP_K(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> torch.Tensor:
    return get_normalized_conditional_entropies(successes, complexities, gen_gaps).min()



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
