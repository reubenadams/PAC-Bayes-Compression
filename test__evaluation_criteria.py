import torch
from scipy import stats

from evaluation_criteria import get_granulated_krccs



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
    result = [krcc1, krcc2]
    expected = get_granulated_krccs(successes, complexities, gen_gaps)
    assert result == expected


def test__get_granulated_krccs__2x3x4():
    shape = (2, 3, 4)
    successes = torch.full(shape, True)
    complexities = torch.rand(shape)
    gen_gaps = torch.rand(shape)

    taus_dim0 = torch.tensor([stats.kendalltau(complexities[:, j, k], gen_gaps[:, j, k]).statistic for j in range(shape[1]) for k in range(shape[2])])
    taus_dim1 = torch.tensor([stats.kendalltau(complexities[i, :, k], gen_gaps[i, :, k]).statistic for i in range(shape[0]) for k in range(shape[2])])
    taus_dim2 = torch.tensor([stats.kendalltau(complexities[i, j, :], gen_gaps[i, j, :]).statistic for i in range(shape[0]) for j in range(shape[1])])
    
    result = [taus_dim0.mean(), taus_dim1.mean(), taus_dim2.mean()]
    expected = get_granulated_krccs(successes, complexities, gen_gaps)
    assert torch.tensor(result).allclose(torch.tensor(expected))


if __name__ == "__main__":
    for _ in range(10):
        test__get_granulated_krccs__2x3()
        test__get_granulated_krccs__2x3x4()