import torch
from scipy.optimize import bisect as sp_bisect


def kl_component(q_i, p_i):
    assert 0 <= q_i <= 1 and 0 <= p_i <= 1
    if q_i == 0:
        return 0
    if p_i == 0:
        return torch.inf
    return q_i * torch.log(q_i / p_i)


def kl_scalars(q, p):
    return kl_component(q, p) + kl_component(1 - q, 1 - p)


def kl_scalars_inverse(q, B, x_tol):
    if B == 0:
        return q
    if q == 0:
        return 1 - torch.exp(-B)  # Easy pen and paper check
    if q == 1:
        return 1
    p_max = 1 - x_tol / 2
    assert q < p_max < 1
    f = lambda p: kl_scalars(q, p) - B
    if f(p_max) < 0:
        print("No upper bound on p")
        return 1
    root = sp_bisect(f=f, a=q, b=p_max, xtol=x_tol)
    return root
