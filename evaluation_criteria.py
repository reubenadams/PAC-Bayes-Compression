from typing import Optional

import torch
from scipy import stats
import numpy as np
from yaml import safe_load
from itertools import product
import pandas as pd


base_hyp_full_names = {
    "hidden_layer_width": "Base Hidden Layer Width",
    "num_hidden_layers": "Base Num Hidden Layers",
    "optimizer_name": "Base Optimizer",
    "batch_size": "Base Batch Size",
    "lr": "Base Learning Rate",
    "dropout_prob": "Base Dropout Prob",
    "weight_decay": "Base Weight Decay",
}


def get_hyp_vals(sweep_config_path: str) -> dict:
    hyp_vals = dict()
    with open(sweep_config_path, "r") as file:
        sweep_config = safe_load(file)
        for name, vals in sweep_config["parameters"].items():
            vals = vals["values"]
            if name not in ["input_dim", "output_dim"]:
                hyp_vals[name] = vals
    return hyp_vals


# TODO: It should take the name of a complexity measure.
def get_sweep_results(hyp_vals: dict, df: pd.DataFrame, complexity_name: Optional[str]) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

    hyp_names = sorted(list(hyp_vals.keys()))  # TODO: Make sure we're always sorting.
    
    results_shape = tuple(len(hyp_vals[name]) for name in hyp_names)  # Number of values for each hyperparameter
    successes = torch.full(results_shape, False)  # To track which runs actually ran and actually met target loss
    if complexity_name is None:
        complexities = None
    else:
        complexities = torch.zeros(results_shape)
    gen_gaps = torch.zeros(results_shape)

    for _, row in df.iterrows():
        run_config = [(name, row[base_hyp_full_names[name]]) for name in hyp_names]
        index = tuple(hyp_vals[name].index(val) for name, val in run_config)
        if row["Base Reached Target"]:
            if complexity_name is not None:
                complexities[index] = row[complexity_name]
            gen_gaps[index] = row["Base Generalization Gap"]
            successes[index] = True
        else:
            print(f"Run {row["run_name"]} with index {index} did not reach target")
            successes[index] = False
    
    return successes, complexities, gen_gaps


def epsilon_oracle(gen_gaps: torch.Tensor, epsilon: float) -> torch.Tensor:
    return gen_gaps + torch.randn(gen_gaps.shape) * epsilon


def get_krcc(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> tuple[float, bool]:
    """Takes three 3x3x3x3x3x3x3 arrays (one dim for each hyperparameter), returns the
    Kendall rank correlation coefficient between complexities and generalization gaps where successes is True"""
    assert successes.shape == complexities.shape == gen_gaps.shape
    complexities = complexities[successes].flatten()
    gen_gaps = gen_gaps[successes].flatten()
    undefined = ((complexities == complexities[0]).all()) or ((gen_gaps == gen_gaps[0]).all())  # KRCC is undefined
    return stats.kendalltau(complexities, gen_gaps).statistic, undefined


def get_oracle_krcc(successes: torch.Tensor, gen_gaps: torch.Tensor, epsilon: float, num_oracle_samples: int) -> float:
    """Takes *two* 3x3x3x3x3x3x3 arrays (one dim for each hyperparameter), generates oracle comlexities, and returns the average
    Kendall rank correlation coefficient between complexities and generalization gaps where successes is True"""
    assert successes.shape == gen_gaps.shape
    num_krccs = 0
    total_krcc = 0
    for _ in range(num_oracle_samples):
        oracle_complexities = epsilon_oracle(gen_gaps=gen_gaps, epsilon=epsilon)
        oracle_krcc, undefined = get_krcc(successes=successes, complexities=oracle_complexities, gen_gaps=gen_gaps)
        if undefined:
            continue
        assert not np.isnan(oracle_krcc)
        num_krccs += 1
        total_krcc += oracle_krcc
    assert num_krccs > 0
    return total_krcc / num_krccs


def get_granulated_krcc_components(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> list[float]:
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
        total_krcc = 0
        for success_batch, comp_batch, gap_batch in zip(successes_flattened, complexities_flattened, gen_gaps_flattened):
            krcc, undefined = get_krcc(success_batch, comp_batch, gap_batch)
            if undefined:
                continue
            assert not np.isnan(krcc)
            num_krccs += 1
            total_krcc += krcc
            krccs_for_dim.append(krcc)
        assert num_krccs > 0
        granulated_krccs.append(total_krcc / num_krccs)
    return granulated_krccs


def get_granulated_krcc(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> float:
    components = get_granulated_krcc_components(successes, complexities, gen_gaps)
    return torch.tensor(components).mean().item()


def get_oracle_granulated_krcc(successes: torch.Tensor, gen_gaps: torch.Tensor, epsilon: float, num_oracle_samples: int) -> float:
    """Takes *two* 3x3x3x3x3x3x3 arrays (one dim for each hyperparameter), generates oracle comlexities, and returns the average *granulated*
    Kendall rank correlation coefficient between complexities and generalization gaps where successes is True"""
    assert successes.shape == gen_gaps.shape
    total_gkrcc = 0
    for _ in range(num_oracle_samples):
        oracle_complexities = epsilon_oracle(gen_gaps=gen_gaps, epsilon=epsilon)
        total_gkrcc += get_granulated_krcc(successes=successes, complexities=oracle_complexities, gen_gaps=gen_gaps)
    return total_gkrcc / num_oracle_samples


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
        for hyp_dim2 in range(hyp_dim1 + 1, num_hyp_dims):
            pmf_joint_triple_Vmu_Vg_US = get_pmf_joint_triple_Vmu_Vg_US(successes, complexities, gen_gaps, hyp_dim1, hyp_dim2)
            pmf_joint_Vg_US = pmf_joint_triple_Vmu_Vg_US.sum(dim=0)
            cond_mut_inf = conditional_mutual_inf(pmf_joint_triple=pmf_joint_triple_Vmu_Vg_US)
            cond_entropy = conditional_entropy(pmf_joint=pmf_joint_Vg_US)
            normed_cond_entropies.append(cond_mut_inf / cond_entropy)
    assert len(normed_cond_entropies) == num_hyp_dims * (num_hyp_dims - 1) // 2
    return torch.tensor(normed_cond_entropies)


def CIT_K(successes: torch.Tensor, complexities: torch.Tensor, gen_gaps: torch.Tensor) -> torch.Tensor:
    return get_normalized_conditional_entropies(successes, complexities, gen_gaps).min()



if __name__ == "__main__":

    complexity_measures = [
        "Inverse Margin Tenth Percentile",
        "Train Loss",
        "Output Entropy",
        "Product Weight Of Fro Norms",
        "Inverse Squared Sigma Ten Percent Increase",
        "KL Bound Sigma Ten Percent Increase",
        "Error Bound Min Over Sigma Inverse KL",
        "Error Bound Min Over Sigma Pinsker",
        "Dist Complexity",
    ]

    oracle_epsilons = [0.02, 0.05]  # Values used in the Fantastic Generalization Measures paper

    hyp_vals = get_hyp_vals("sweep_config_comb_toy.yaml")
    combined_df = pd.read_csv(r"distillation\models\MNIST1D\40\dist_metrics\combined.csv")

    # for complexity_measure in complexity_measures:
    #     print(complexity_measure)

    #     successes, complexities, gen_gaps = get_sweep_results(hyp_vals=hyp_vals, df=combined_df, complexity_name=complexity_measure)
    #     print(f"\t{successes.shape=}, {complexities.shape=}, {gen_gaps.shape=}")
    #     print(f"\tNum successes = {(successes == True).sum()}")
    #     print(f"\tNum failures = {(successes == False).sum()}")

    #     krcc = get_krcc(successes=successes, complexities=complexities, gen_gaps=gen_gaps)
    #     print(f"\tKRCC: {krcc}")

    #     # granulated_krccs = get_granulated_krcc_components(successes=successes, complexities=gen_gaps, gen_gaps=gen_gaps)
    #     # print(granulated_krccs)
    #     # granulated_krcc = get_granulated_krcc(successes, complexities=gen_gaps, gen_gaps=gen_gaps)
    #     # print(granulated_krcc)
    #     print(CIT_K(successes=successes, complexities=gen_gaps, gen_gaps=gen_gaps))
    #     print()
    
    for epsilon in oracle_epsilons:
        successes, _, gen_gaps = get_sweep_results(hyp_vals=hyp_vals, df=combined_df, complexity_name=None)
        print(f"\t{successes.shape=}, {gen_gaps.shape=}")
        print(f"\tNum successes = {(successes == True).sum()}")
        print(f"\tNum failures = {(successes == False).sum()}")
        oracle_krcc = get_oracle_krcc(successes=successes, gen_gaps=gen_gaps, epsilon=epsilon, num_oracle_samples=10000)
        print(f"\tOracle KRCC: {oracle_krcc}")

        oracle_granulated_krcc = get_oracle_granulated_krcc(successes=successes, gen_gaps=gen_gaps, epsilon=epsilon, num_oracle_samples=10000)
        print(f"\tOracle Granulated KRCC: {oracle_granulated_krcc}")

