import torch
from scipy import stats

from evaluation_criteria import get_gkrcc_components, entropy, conditional_entropy, mutual_inf, conditional_mutual_inf, get_differences, get_signs, get_joint_probs_two_hyp_dims, get_joint_probs_two_hyp_dims



def test__get_granulated_krccs__2x3():
    shape = (2, 3)
    successes = torch.full(shape, True)
    complexities = torch.rand(shape)
    gen_gaps = torch.rand(shape)

    tau_row1 = stats.kendalltau(complexities[0], gen_gaps[0]).statistic
    tau_row2 = stats.kendalltau(complexities[1], gen_gaps[1]).statistic

    tau_col1 = stats.kendalltau(complexities[:, 0], gen_gaps[:, 0]).statistic
    tau_col2 = stats.kendalltau(complexities[:, 1], gen_gaps[:, 1]).statistic
    tau_col3 = stats.kendalltau(complexities[:, 2], gen_gaps[:, 2]).statistic

    krcc1 = (tau_col1 + tau_col2 + tau_col3) / 3
    krcc2 = (tau_row1 + tau_row2) / 2
    expected = [krcc1, krcc2]
    result = get_gkrcc_components(successes, complexities, gen_gaps)
    assert result == expected


def test__get_granulated_krccs__2x3x4():
    shape = (2, 3, 4)
    successes = torch.full(shape, True)
    complexities = torch.rand(shape)
    gen_gaps = torch.rand(shape)

    taus_dim0 = torch.tensor([stats.kendalltau(complexities[:, j, k], gen_gaps[:, j, k]).statistic for j in range(shape[1]) for k in range(shape[2])])
    taus_dim1 = torch.tensor([stats.kendalltau(complexities[i, :, k], gen_gaps[i, :, k]).statistic for i in range(shape[0]) for k in range(shape[2])])
    taus_dim2 = torch.tensor([stats.kendalltau(complexities[i, j, :], gen_gaps[i, j, :]).statistic for i in range(shape[0]) for j in range(shape[1])])
    
    expected = [taus_dim0.mean(), taus_dim1.mean(), taus_dim2.mean()]
    result = get_gkrcc_components(successes, complexities, gen_gaps)
    assert torch.tensor(result).allclose(torch.tensor(expected))


def test__entropy__uniform():
    k = 100
    pmf = torch.ones(k) / k
    expected = torch.log(torch.tensor(k))
    result = entropy(pmf)
    assert torch.isclose(result, expected)


def test__entropy__uniform_with_zeros():
    n = 100
    pmf = torch.randint(0, 2, (n,))
    while torch.all(pmf == 0):
        pmf = torch.randint(0, 2, (n,))
    k = pmf.sum()
    pmf = pmf / k
    assert pmf.sum() == 1, f"pmf.sum()={pmf.sum()}"
    expected = torch.log(k)
    result = entropy(pmf)
    assert torch.isclose(result, expected)


def test__mutual_inf__2x2():
    n1, n2 = 2, 2
    pmf_joint = torch.rand((n1, n2))
    pmf_joint = pmf_joint / pmf_joint

    pmf0 = pmf_joint.sum(1)
    pmf1 = pmf_joint.sum(0)

    term00 = pmf_joint[0, 0] * torch.log(pmf_joint[0, 0] / (pmf0[0] * pmf1[0]))
    term01 = pmf_joint[0, 1] * torch.log(pmf_joint[0, 1] / (pmf0[0] * pmf1[1]))
    term10 = pmf_joint[1, 0] * torch.log(pmf_joint[1, 0] / (pmf0[1] * pmf1[0]))
    term11 = pmf_joint[1, 1] * torch.log(pmf_joint[1, 1] / (pmf0[1] * pmf1[1]))

    expected = term00 + term01 + term10 + term11
    result = mutual_inf(pmf_joint)
    assert torch.isclose(result, expected)


def test__mutual_inf__big():
    n1, n2 = 100, 201
    pmf_joint = torch.rand((n1, n2))
    pmf_joint = pmf_joint / pmf_joint

    pmf0 = pmf_joint.sum(1)
    pmf1 = pmf_joint.sum(0)

    terms = [pmf_joint[i, j] * torch.log(pmf_joint[i, j] / (pmf0[i] * pmf1[j])) for i in range(n1) for j in range(n2)]
    expected = torch.tensor(terms).sum()
    result = mutual_inf(pmf_joint)
    assert torch.isclose(result, expected)


def test__mutual_inf__big__with_zeros():
    n1, n2 = 132, 623
    pmf_joint = torch.rand((n1, n2))
    mask = torch.randint(0, 2, (n1, n2)) == 1
    pmf_joint[mask] = 0 
    pmf_joint = pmf_joint / pmf_joint.sum()

    pmf0 = pmf_joint.sum(1)
    pmf1 = pmf_joint.sum(0)

    terms = [pmf_joint[i, j] * torch.log(pmf_joint[i, j] / (pmf0[i] * pmf1[j])) for i in range(n1) for j in range(n2) if pmf_joint[i, j] != 0]
    expected = torch.tensor(terms).sum()
    result = mutual_inf(pmf_joint)
    assert torch.isclose(result, expected)


def test__conditional_entropy():
    n1, n2 = 132, 623
    pmf_joint = torch.rand((n1, n2))
    mask = torch.randint(0, 2, (n1, n2)) == 1
    pmf_joint[mask] = 0 
    pmf_joint = pmf_joint / pmf_joint.sum()

    pmf_y = pmf_joint.sum(0)

    expected = 0
    for i in range(n1):
        for j in range(n2):
            p_xy = pmf_joint[i, j]
            p_y = pmf_y[j]
            if p_xy == 0 or p_y == 0:
                continue
            p_x_given_y = p_xy / p_y
            expected -= p_y * p_x_given_y * torch.log(p_x_given_y)
    result = conditional_entropy(pmf_joint)
    assert torch.isclose(result, expected)


def test__conditional_mutual_inf():
    n1, n2, n3 = 13, 63, 41
    pmf_joint_triple = torch.rand((n1, n2, n3))
    mask = torch.randint(0, 2, (n1, n2, n3)) == 1
    pmf_joint_triple[mask] = 0 
    pmf_joint_triple = pmf_joint_triple / pmf_joint_triple.sum()

    pmf_z = pmf_joint_triple.sum(0).sum(0)
    pmf_xz = pmf_joint_triple.sum(1)
    pmf_yz = pmf_joint_triple.sum(0)

    expected = 0
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                p_z = pmf_z[k]
                p_xyz = pmf_joint_triple[i, j, k]
                p_xz = pmf_xz[i, k]
                p_yz = pmf_yz[j, k]
                if p_z == 0 or p_xyz == 0 or p_xz == 0 or p_yz == 0:
                    continue
                p_xy_given_z = p_xyz / p_z
                p_x_given_z = p_xz / p_z
                p_y_given_z = p_yz / p_z
                expected += p_z * p_xy_given_z * torch.log(p_xy_given_z / (p_x_given_z * p_y_given_z))
    result = conditional_mutual_inf(pmf_joint_triple)
    assert torch.isclose(result, expected)


def test__get_differences():
    a = torch.tensor([1, 2, 2, 3])
    expected = torch.tensor([-1, -1, -2, 1, 0, -1, 1, 0, -1, 2, 1, 1])
    result = get_differences(a)
    assert (result == expected).all()


def test__get_signs():
    a = torch.tensor([1, 2, 2, 3])
    expected = torch.tensor([-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1])
    result = get_signs(a)
    assert (result == expected).all()


def test__get_joint_probs():
    complexities_signs = torch.tensor([1,  1,  1,  1, -1, -1, -1, -1])
    gen_gaps_signs =     torch.tensor([1, -1, -1, -1, -1, -1,  1, -1])
    prob_hyp1_hyp2 = torch.tensor(1/9)
    prob_nn = 3/8
    prob_np = 1/8
    prob_pn = 3/8
    prob_pp = 1/8
    expected = tuple(p * prob_hyp1_hyp2 for p in [prob_nn, prob_np, prob_pn, prob_pp])
    result = get_joint_probs_two_hyp_dims(complexities_signs, gen_gaps_signs, prob_hyp1_hyp2)
    assert result == expected, f"{result=}, {expected=}"


def test__get_joint_probs():
    successes = torch.tensor([[True, True, False], [True, True, False]])
    complexities = torch.tensor([[2, 4, 100], [8, 16, 100]])
    gen_gaps = torch.tensor([[0.1, 0.3, 100], [0.2, 0.01, 100]])

    successes = successes.reshape(2, 1, 3, 1)
    complexities = complexities.reshape(2, 1, 3, 1)
    gen_gaps = gen_gaps.reshape(2, 1, 3, 1)

    successes = torch.tile(successes, (1, 7, 1, 5))
    complexities = torch.tile(complexities, (1, 7, 1, 5))
    gen_gaps = torch.tile(gen_gaps, (1, 7, 1, 5))

    slices = [slice(None), 0, slice(None), 0]
    prob_hyp1_hyp2 = torch.tensor(1/9)

    complexities_signs = torch.tensor([-1, -1, -1, 1, -1, -1, 1,  1, -1,  1,  1,  1])
    gen_gaps_signs =     torch.tensor([-1, -1,  1, 1,  1,  1, 1, -1,  1, -1, -1, -1])
    count_nn = 2 / 12
    count_np = 4 / 12
    count_pn = 4 / 12
    count_pp = 2 / 12
    
    expected = tuple(p * prob_hyp1_hyp2 for p in [count_nn, count_np, count_pn, count_pp])
    result = get_joint_probs_two_hyp_dims(successes, complexities, gen_gaps, slices, prob_hyp1_hyp2)
    assert result == expected
    






if __name__ == "__main__":
    for _ in range(10):
        test__get_differences()
        test__get_signs()
        test__get_joint_probs()
        test__get_granulated_krccs__2x3()
        test__get_granulated_krccs__2x3x4()
    test__entropy__uniform()
    test__entropy__uniform_with_zeros()
    test__mutual_inf__2x2()
    test__mutual_inf__big()
    test__mutual_inf__big__with_zeros()
    test__conditional_entropy()
    test__conditional_mutual_inf()
