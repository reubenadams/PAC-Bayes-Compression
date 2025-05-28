import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import ComplexityMeasures


def main():
    target_CE_loss_increase = 0.1
    combined_df = pd.read_csv(r"distillation\models\MNIST1D\40\all_metrics\combined.csv")
    complexity_measure_names = ComplexityMeasures.get_all_names(target_CE_loss_increase=target_CE_loss_increase)

    print(len(complexity_measure_names), "complexity measures found:")
    y = combined_df["Base Generalization Gap"]
    for name in complexity_measure_names:
        plt.figure()
        matplotlib_name = ComplexityMeasures.get_matplotlib_name(name, target_CE_loss_increase=target_CE_loss_increase)
        x = combined_df[name]
        plt.scatter(x, y, s=2)
        plt.xlabel(matplotlib_name, fontsize=20)
        plt.ylabel(r"Generalization Gap, $R_D(h_{W, B})-R_S(h_{W, B})$", fontsize=14)
        if ComplexityMeasures.use_log_x_axis(name):
            plt.xscale("log")
        plt.tight_layout()
        plt.savefig(fr"distillation\models\MNIST1D\40\figures\measures\{name}_vs_generalization_gap.png", dpi=300)
        plt.close()
        # plt.show()


def plot_dist_against_size():
    
    df = pd.read_csv(r"distillation\models\MNIST1D\40\all_metrics\combined.csv")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create violin plot
    sns.violinplot(data=df, x="Base Num Hidden Layers", y="Dist Complexity", 
                   ax=ax, color='lightblue', alpha=0.7, inner="quartile")
    
    # Add strip plot on top
    sns.stripplot(data=df, x="Base Num Hidden Layers", y="Dist Complexity", 
                  ax=ax, color='darkblue', alpha=0.6, size=4, jitter=True)
    
    plt.grid(True, alpha=0.3)
    ax.set_xlabel(r'$\mu_\text{dist-complexity}$', fontsize=16)
    ax.set_ylabel("Base Num Hidden Layers", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(r"distillation\models\MNIST1D\40\figures\dist_vs_size\dist_vs_num_hidden_layers.png", dpi=300)
    plt.show()
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create violin plot
    sns.violinplot(data=df, x="Base Hidden Layer Width", y="Dist Complexity", 
                   ax=ax, color='lightblue', alpha=0.7, inner="quartile")
    
    # Add strip plot on top
    sns.stripplot(data=df, x="Base Hidden Layer Width", y="Dist Complexity", 
                  ax=ax, color='darkblue', alpha=0.6, size=4, jitter=True)
    
    plt.grid(True, alpha=0.3)
    ax.set_xlabel(r'$\mu_\text{dist-complexity}$', fontsize=16)
    ax.set_ylabel("Base Hidden Layer Width", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(r"distillation\models\MNIST1D\40\figures\dist_vs_size\dist_vs_hidden_width.png", dpi=300)
    plt.show()
    plt.close()




def plot_dist_vs_size():
    df = pd.read_csv(r"distillation\models\MNIST1D\40\all_metrics\combined.csv")
    num_layers = df["Base Num Hidden Layers"].unique()
    hidden_widths = df["Base Hidden Layer Width"].unique()
    num_params = {(layers, width): get_num_params(layers, width) for layers in num_layers for width in hidden_widths}
    df["Base Number of Params"] = df.apply(lambda row: num_params[(row["Base Num Hidden Layers"], row["Base Hidden Layer Width"])], axis=1)
    
    # Convert to millions
    df["Base Number of Params (Millions)"] = (df["Base Number of Params"] / 1e6).round(2)

    print(num_params.values())
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create violin plot
    sns.violinplot(data=df, x="Base Number of Params (Millions)", y="Dist Complexity", 
                   ax=ax, color='lightblue', alpha=0.7)
    
    # Add strip plot on top
    sns.stripplot(data=df, x="Base Number of Params (Millions)", y="Dist Complexity", 
                  ax=ax, color='darkblue', alpha=0.6, size=4, jitter=True)
    
    plt.grid(True, alpha=0.3)
    ax.set_ylabel(r'$\mu_\text{dist-complexity}$', fontsize=16)
    ax.set_xlabel("Number of Parameters (Millions)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(r"distillation\models\MNIST1D\40\figures\dist_vs_size\dist_vs_num_params.png", dpi=300)
    plt.show()
    plt.close()


def get_num_params(num_hidden_layers, hidden_layer_width):
    """
    Calculate the number of parameters in a fully connected neural network with bias terms.
    """
    dimensions = [784] + [hidden_layer_width] * num_hidden_layers + [10]
    num_params = 0
    for i in range(len(dimensions) - 1):
        num_params += dimensions[i] * dimensions[i + 1]  # weights
        num_params += dimensions[i + 1]  # biases
    return num_params


if __name__ == "__main__":
    # main()
    plot_dist_against_size()
    plot_dist_vs_size()