import torch
from scipy.optimize import bisect


def kl_component(q_i, p_i):
    assert 0 <= q_i <= 1 and 0 <= p_i <= 1
    if q_i == 0:
        return 0
    if p_i == 0:
        return torch.inf
    return q_i * torch.log(q_i / p_i)


def kl_scalars(q, p):
    return kl_component(q, p) + kl_component(1 - q, 1 - p)


def kl_scalars_inverse(q, B, x_tol=2e-12):
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
    root = bisect(f=f, a=q, b=p_max, xtol=x_tol)
    return root


def pacb_kl_bound(KL, n, delta):
    return (KL + torch.log(2 * torch.sqrt(torch.tensor(n)) / delta)) / n


def pacb_error_bound_inverse_kl(empirical_error, KL, n, delta):
    kl_bound = pacb_kl_bound(KL=KL, n=n, delta=delta)
    return kl_scalars_inverse(q=empirical_error, B=kl_bound)


def pacb_error_bound_pinsker(empirical_error, KL, n, delta):
    kl_bound = pacb_kl_bound(KL=KL, n=n, delta=delta)
    return empirical_error + torch.sqrt(kl_bound / 2)
