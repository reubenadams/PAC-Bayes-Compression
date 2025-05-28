import json
import numpy as np
import matplotlib.pyplot as plt


def get_oracle_means_and_stds():

    with open(r"distillation\models\MNIST1D\40\evaluation_metrics\oracle\all_oracle_evaluation_metrics.json", "r") as f:
        data = json.load(f)

    std_props = list(dict.fromkeys(metrics["std_proportion"] for metrics in data))
    epsilons = list(dict.fromkeys(metrics["epsilon"] for metrics in data))
    collated_metrics = []

    for std_prop, epsilon in zip(std_props, epsilons):
        metrics = {
            "std_proportion": std_prop,
            "epsilon": epsilon,
        }
        r_values = []
        r_squared_values = []
        krcc_values = []
        gkrcc_values = []
        cit_values = []
        for metric in data:
            if metric["std_proportion"] == std_prop and metric["epsilon"] == epsilon:
                r_values.append(metric["R Value"])
                r_squared_values.append(metric["R Squared"])
                krcc_values.append(metric["KRCC"])
                gkrcc_values.append(metric["GKRCC"])
                cit_values.append(metric["CIT At Most Two Hyp Dims"])
        metrics["R Value Mean"] = np.mean(r_values)
        metrics["R Value Std"] = np.std(r_values)
        metrics["R Squared Mean"] = np.mean(r_squared_values)
        metrics["R Squared Std"] = np.std(r_squared_values)
        metrics["KRCC Mean"] = np.mean(krcc_values)
        metrics["KRCC Std"] = np.std(krcc_values)
        metrics["GKRCC Mean"] = np.mean(gkrcc_values)
        metrics["GKRCC Std"] = np.std(gkrcc_values)
        metrics["CIT Mean"] = np.mean(cit_values)
        metrics["CIT Std"] = np.std(cit_values)
        collated_metrics.append(metrics)

    with open(r"distillation\models\MNIST1D\40\evaluation_metrics\oracle\collated_oracle_evaluation_metrics.json", "w") as f:
        json.dump(collated_metrics, f, indent=2)


def plot_oracle_means_and_stds():

    with open(r"distillation\models\MNIST1D\40\evaluation_metrics\oracle\collated_oracle_evaluation_metrics.json", "r") as f:
        data = json.load(f)

    std_props = [metrics["std_proportion"] for metrics in data]
    epsilons = [metrics["epsilon"] for metrics in data]

    r_means = [metrics["R Value Mean"] for metrics in data]
    r_stds = [metrics["R Value Std"] for metrics in data]

    r_squared_means = [metrics["R Squared Mean"] for metrics in data]
    r_squared_stds = [metrics["R Squared Std"] for metrics in data]

    krcc_means = [metrics["KRCC Mean"] for metrics in data]
    krcc_stds = [metrics["KRCC Std"] for metrics in data]

    gkrcc_means = [metrics["GKRCC Mean"] for metrics in data]
    gkrcc_stds = [metrics["GKRCC Std"] for metrics in data]

    cit_means = [metrics["CIT Mean"] for metrics in data]
    cit_stds = [metrics["CIT Std"] for metrics in data]

    # Create figure and first axis
    fig, ax1 = plt.subplots()

    # Plot data using std_props on bottom x-axis
    ax1.errorbar(std_props, r_means, yerr=r_stds, fmt='-', markersize=5, label=r'$\rho$, Correlation Coefficient')
    ax1.errorbar(std_props, r_squared_means, yerr=r_squared_stds, fmt='-', markersize=5, label=r'$R^2$, Explained Variance')
    ax1.errorbar(std_props, krcc_means, yerr=krcc_stds, fmt='-', markersize=5, label=r'$\tau$, KRCC')
    ax1.errorbar(std_props, gkrcc_means, yerr=gkrcc_stds, fmt='-', markersize=5, label=r'$\Psi$, GKRCC')
    ax1.errorbar(std_props, cit_means, yerr=cit_stds, fmt='-', markersize=5, label=r'$\mathcal{K}$, CIT')

    # Configure bottom x-axis (std_props)
    ax1.set_xlabel(r'Oracle noise as proportion of $\sigma_{\text{gen-gaps}}$')
    ax1.set_xscale('log')

    # Create second x-axis for epsilon
    ax2 = ax1.twiny()
    ax2.set_xlabel(r'Oracle noise $\epsilon$')
    ax2.set_xscale('log')
    
    # Let matplotlib auto-scale both axes with natural padding
    # Then ensure they're aligned by matching the limits
    ax1_xlim = ax1.get_xlim()
    ax2.set_xlim(ax1_xlim[0] * min(epsilons) / min(std_props), 
                 ax1_xlim[1] * max(epsilons) / max(std_props))

    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig(r"distillation\models\MNIST1D\40\figures\metrics\oracle_metrics.png", dpi=300)


if __name__ == "__main__":
    get_oracle_means_and_stds()
    print("Collated oracle evaluation metrics saved.")
    plot_oracle_means_and_stds()