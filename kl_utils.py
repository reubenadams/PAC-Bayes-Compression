import torch
from scipy.optimize import bisect
import torch.nn.functional as F


def kl_component(q_i, p_i):
    assert 0 <= q_i <= 1 and 0 <= p_i <= 1
    if q_i == 0:
        return torch.tensor(0.)
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
        return torch.tensor(1.)
    p_max = 1 - x_tol / 2
    assert q < p_max < 1
    f = lambda p: kl_scalars(q, p) - B
    if f(p_max) < 0:
        return torch.tensor(1.)
    root = bisect(f=f, a=q, b=p_max, xtol=x_tol)
    return torch.tensor(root)


def distillation_loss(
        teacher_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the knowledge distillation loss between teacher and student, averaged over the batch. Equals KL(teacher || student)."""
    if not (teacher_probs.shape == student_log_probs.shape == teacher_log_probs.shape):
        raise ValueError("All input tensors must have the same shape.")
    if teacher_probs.min() < 0 or teacher_probs.max() > 1:
        raise ValueError("Student probabilities must be in the range [0, 1].")
    kls = teacher_probs * (teacher_log_probs - student_log_probs)
    return kls.sum(dim=-1).mean()


def pacb_kl_bound(KL, n, delta):
    return (KL + torch.log(2 * torch.sqrt(torch.tensor(n)) / delta)) / n


def pacb_error_bound_inverse_kl(empirical_error, KL, n, delta):
    kl_bound = pacb_kl_bound(KL=KL, n=n, delta=delta)
    return kl_scalars_inverse(q=empirical_error, B=kl_bound)


def pacb_error_bound_pinsker(empirical_error, KL, n, delta):
    kl_bound = pacb_kl_bound(KL=KL, n=n, delta=delta)
    return empirical_error + torch.sqrt(kl_bound / 2)
