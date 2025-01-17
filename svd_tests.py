import torch
import matplotlib.pyplot as plt


d = 128
weight = torch.randn(784, 128)
weight_norm = torch.linalg.norm(weight, ord=2)
norms = []
U, S, Vt = torch.linalg.svd(weight)
for rank in range(1, d + 1):
    U_low_rank = U[:, :rank]
    S_low_rank = S[:rank]
    Vt_low_rank = Vt[:rank, :]
    weight_approx = U_low_rank @ torch.diag(S_low_rank) @ Vt_low_rank
    gap = weight_approx - weight
    spectral_norm = torch.linalg.norm(gap, ord=2)
    norms.append(spectral_norm)
    print(f"Rank: {rank}, Norm: {spectral_norm}")
plt.plot(range(1, d + 1), norms)
plt.hlines([weight_norm, weight_norm / 2], 1, d, colors='r', linestyles='dashed')
plt.show()
