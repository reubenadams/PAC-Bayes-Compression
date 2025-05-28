import json
import matplotlib.pyplot as plt


with open(r"distillation\models\MNIST1D\40\evaluation_metrics\all_evaluation_metrics.json", "r") as f:
    data = json.load(f)


names = list(data.keys())
names[-1] += " (Ours)"
r_values = [metrics["R Value"] for metrics in data.values()]
r_squared_values = [metrics["R Squared"] for metrics in data.values()]
krcc_values = [metrics["KRCC"] for metrics in data.values()]
gkrcc_values = [metrics["GKRCC"] for metrics in data.values()]
cit_values = [metrics["CIT At Most Two Hyp Dims"] for metrics in data.values()]


plt.figure(figsize=(8, 6))
# Set x-axis labels
plt.xticks(range(len(names)), names, rotation=90, ha='center')
ax = plt.gca()
labels = ax.get_xticklabels()
labels[-1].set_fontweight('bold')  # Make the last label bold
plt.ylim(-0.9, 0.9)

plt.scatter(range(len(names)), r_values, label=r'$\rho$, Correlation Coefficient', s=10)
plt.scatter(range(len(names)), r_squared_values, label=r'$R^2$, Explained Variance', s=10)
plt.scatter(range(len(names)), krcc_values, label=r'$\tau$, KRCC', s=10)
plt.scatter(range(len(names)), gkrcc_values, label=r'$\Psi$, GKRCC', s=10)
plt.scatter(range(len(names)), cit_values, label=r'$\mathcal{K}$, CIT', s=10)
plt.legend()
plt.tight_layout()
plt.savefig(r"distillation\models\MNIST1D\40\figures\metrics\all_evaluation_metrics_plot.png", dpi=300)
