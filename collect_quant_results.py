import os
from config import FinalCompResults
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


# Set global font sizes
plt.rcParams.update({
    'font.size': 12,             # Base font size
    'axes.titlesize': 16,        # Title font size
    'axes.labelsize': 14,        # X and Y labels font size
    'xtick.labelsize': 10,       # X tick labels size
    'ytick.labelsize': 10,       # Y tick labels size
    'legend.fontsize': 12,       # Legend font size
    'figure.titlesize': 18       # Figure title size
})


exponent_colors = plt.cm.viridis(np.linspace(0, 1, 9))


comp_schemes = {
    "no_comp": "No Comp",
    "quant_k_means": "K-Means",
    "quant_trunc": "Trunc",
    "low_rank": "Low Rank",
    "low_rank_and_quant_k_means": "Low Rank + K-Means",
    "low_rank_and_quant_trunc": "Low Rank + Trunc",
}


def get_filepath(
        dataset_name: str,
        hw: int,
        nl: int,
        comp_scheme: str,
):
    filepath = os.path.join(
        "quantization",
        "cluster_results",
        dataset_name,
        f"{comp_scheme}_metrics",
        f"opadam_hw{hw}_nl{nl}_lr0.001_bs128_dp0_wd0.json",
    )
    return filepath



def plot_trunc_results(
        dataset_name: str,
        hw: int,
        nl: int,
):
    no_comp_filepath = get_filepath(dataset_name, hw, nl, "no_comp")
    comp_filepath = get_filepath(dataset_name, hw, nl, "quant_trunc")

    try:
        no_comp_results = FinalCompResults.load_from_json(filepath=no_comp_filepath).best_results
        all_comp_results = FinalCompResults.load_from_json(filepath=comp_filepath).all_results
    except FileNotFoundError:
        print(f"File not found: {no_comp_filepath} or {comp_filepath}. Skipping.")
        return

    no_comp_error = 1 - no_comp_results.train_accuracy
    no_comp_string_length = no_comp_results.KL / sqrt(2)
    no_comp_inverse_kl_bound = no_comp_results.error_bound_inverse_kl_spectral_domain
    no_comp_pinsker_bound = no_comp_results.error_bound_pinsker_spectral_domain

    exponent_range = list(range(9))

    # Create a 2x2 subplot figure with shared x-axis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots

    # Plot 1: Train Errors (top-left)
    for exponent_bits in exponent_range:
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
        comp_train_errors = [1 - results.train_accuracy for results in all_comp_results if results.exponent_bits == exponent_bits]
        axes[0, 0].plot(comp_string_lengths, comp_train_errors, marker='o', markersize=3, color=exponent_colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[0, 0].axhline(y=no_comp_error, color='k', linestyle='--', label="No Compression")
    axes[0, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 0].set_ylabel("Error on train set")
    axes[0, 0].set_ylim(0.25, 1.05)

    # Plot 2: Margin Losses (top-right)
    for exponent_bits in exponent_range:
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
        comp_margin_losses = [results.train_margin_loss_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
        axes[0, 1].plot(comp_string_lengths, comp_margin_losses, marker='o', markersize=3, color=exponent_colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[0, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 1].set_ylabel("Margin loss on train set")
    axes[0, 1].set_ylim(0.25, 1.05)

    # Plot 3: Inverse KL Bounds (bottom-left)
    for exponent_bits in exponent_range:
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
        comp_inverse_kl_bounds = [results.error_bound_inverse_kl_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
        axes[1, 0].plot(comp_string_lengths, comp_inverse_kl_bounds, marker='o', markersize=3, color=exponent_colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[1, 0].axhline(y=no_comp_inverse_kl_bound, color='k', linestyle='--')
    axes[1, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 0].set_xlabel("String length")
    axes[1, 0].set_ylabel("Error bound, inverse kl")
    axes[1, 0].set_ylim(0.8625, 1.0125)

    # Plot 4: Pinsker Bounds (bottom-right)
    for exponent_bits in exponent_range:
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
        comp_pinsker_bounds = [results.error_bound_pinsker_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
        axes[1, 1].plot(comp_string_lengths, comp_pinsker_bounds, marker='o', markersize=3, color=exponent_colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[1, 1].axhline(y=no_comp_pinsker_bound, color='k', linestyle='--')
    axes[1, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 1].set_xlabel("String length")
    axes[1, 1].set_ylabel("Error bound, Pinsker")
    axes[1, 1].set_ylim(0.85, 1.85)

    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
            ncol=5, fancybox=True, shadow=True)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plot_path = comp_filepath[:-5] + ".png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.show()


def plot_k_means_results(
        dataset_name: str,
        hw: int,
        nl: int,
):
    no_comp_filepath = get_filepath(dataset_name, hw, nl, "no_comp")
    comp_filepath = get_filepath(dataset_name, hw, nl, "quant_k_means")

    try:
        no_comp_results = FinalCompResults.load_from_json(filepath=no_comp_filepath).best_results
        all_comp_results = FinalCompResults.load_from_json(filepath=comp_filepath).all_results
    except FileNotFoundError:
        print(f"File not found: {no_comp_filepath} or {comp_filepath}. Skipping.")
        return

    no_comp_error = 1 - no_comp_results.train_accuracy
    no_comp_string_length = no_comp_results.KL / sqrt(2)
    no_comp_inverse_kl_bound = no_comp_results.error_bound_inverse_kl_spectral_domain
    no_comp_pinsker_bound = no_comp_results.error_bound_pinsker_spectral_domain

    # Create a 2x2 subplot figure with shared x-axis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots

    # Plot 1: Train Errors (top-left)
    comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results]
    comp_train_errors = [1 - results.train_accuracy for results in all_comp_results]
    axes[0, 0].plot(comp_string_lengths, comp_train_errors, marker='o', markersize=3, color=exponent_colors[4], label="K-Means")
    axes[0, 0].axhline(y=no_comp_error, color='k', linestyle='--', label="No Compression")
    axes[0, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 0].set_ylabel("Error on train set")
    axes[0, 0].set_ylim(0.25, 1.05)

    # Plot 2: Margin Losses (top-right)
    comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results]
    comp_margin_losses = [results.train_margin_loss_spectral_domain for results in all_comp_results]
    axes[0, 1].plot(comp_string_lengths, comp_margin_losses, marker='o', markersize=3, color=exponent_colors[4])
    axes[0, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 1].set_ylabel("Margin loss on train set")
    axes[0, 1].set_ylim(0.25, 1.05)

    # Plot 3: Inverse KL Bounds (bottom-left)
    comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results]
    comp_inverse_kl_bounds = [results.error_bound_inverse_kl_spectral_domain for results in all_comp_results]
    axes[1, 0].plot(comp_string_lengths, comp_inverse_kl_bounds, marker='o', markersize=3, color=exponent_colors[4])
    axes[1, 0].axhline(y=no_comp_inverse_kl_bound, color='k', linestyle='--')
    axes[1, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 0].set_xlabel("String length")
    axes[1, 0].set_ylabel("Error bound, inverse kl")
    axes[1, 0].set_ylim(0.8625, 1.0125)

    # Plot 4: Pinsker Bounds (bottom-right)
    comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results]
    comp_pinsker_bounds = [results.error_bound_pinsker_spectral_domain for results in all_comp_results]
    axes[1, 1].plot(comp_string_lengths, comp_pinsker_bounds, marker='o', markersize=3, color=exponent_colors[4])
    axes[1, 1].axhline(y=no_comp_pinsker_bound, color='k', linestyle='--')
    axes[1, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 1].set_xlabel("String length")
    axes[1, 1].set_ylabel("Error bound, Pinsker")
    axes[1, 1].set_ylim(0.85, 1.85)

    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
            ncol=2, fancybox=True, shadow=True)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plot_path = comp_filepath[:-5] + ".png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.show()


def plot_low_rank_results(
        dataset_name: str,
        hw: int,
        nl: int,
):
    
    assert nl == 1, "Plot looks best for 1 hidden layers"

    no_comp_filepath = get_filepath(dataset_name, hw, nl, "no_comp")
    comp_filepath = get_filepath(dataset_name, hw, nl, "low_rank")

    try:
        no_comp_results = FinalCompResults.load_from_json(filepath=no_comp_filepath).best_results
        all_comp_results = FinalCompResults.load_from_json(filepath=comp_filepath).all_results
    except FileNotFoundError:
        print(f"File not found: {no_comp_filepath} or {comp_filepath}. Skipping.")
        return

    no_comp_error = 1 - no_comp_results.train_accuracy
    no_comp_string_length = no_comp_results.KL / sqrt(2)
    no_comp_inverse_kl_bound = no_comp_results.error_bound_inverse_kl_spectral_domain
    no_comp_pinsker_bound = no_comp_results.error_bound_pinsker_spectral_domain

    r1_vals = sorted(list(set(results.ranks[0] for results in all_comp_results)))
    r1_colors = plt.cm.viridis(np.linspace(0, 1, len(r1_vals)))

    # Create a 2x2 subplot figure with shared x-axis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots

    # Plot 1: Train Errors (top-left)
    for idx, r1 in enumerate(r1_vals):
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.ranks[0] == r1]
        comp_train_errors = [1 - results.train_accuracy for results in all_comp_results if results.ranks[0] == r1]
        axes[0, 0].plot(comp_string_lengths, comp_train_errors, marker='o', markersize=3, color=r1_colors[idx], label=f"$r_1 = {r1}$")
    axes[0, 0].axhline(y=no_comp_error, color='k', linestyle='--', label="No Compression")
    axes[0, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 0].set_ylabel("Error on train set")
    axes[0, 0].set_ylim(0.25, 1.05)

    # Plot 2: Margin Losses (top-right)
    for idx, r1 in enumerate(r1_vals):
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.ranks[0] == r1]
        comp_margin_losses = [results.train_margin_loss_spectral_domain for results in all_comp_results if results.ranks[0] == r1]
        axes[0, 1].plot(comp_string_lengths, comp_margin_losses, marker='o', markersize=3, color=r1_colors[idx], label=f"$r_1 = {r1}$")
    axes[0, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 1].set_ylabel("Margin loss on train set")
    axes[0, 1].set_ylim(0.25, 1.05)

    # Plot 3: Inverse KL Bounds (bottom-left)
    for idx, r1 in enumerate(r1_vals):
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.ranks[0] == r1]
        comp_inverse_kl_bounds = [results.error_bound_inverse_kl_spectral_domain for results in all_comp_results if results.ranks[0] == r1]
        axes[1, 0].plot(comp_string_lengths, comp_inverse_kl_bounds, marker='o', markersize=3, color=r1_colors[idx], label=f"$r_1 = {r1}$")
    axes[1, 0].axhline(y=no_comp_inverse_kl_bound, color='k', linestyle='--')
    axes[1, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 0].set_xlabel("String length")
    axes[1, 0].set_ylabel("Error bound, inverse kl")
    axes[1, 0].set_ylim(0.8625, 1.0125)

    # Plot 4: Pinsker Bounds (bottom-right)
    for idx, r1 in enumerate(r1_vals):
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.ranks[0] == r1]
        comp_pinsker_bounds = [results.error_bound_pinsker_spectral_domain for results in all_comp_results if results.ranks[0] == r1]
        axes[1, 1].plot(comp_string_lengths, comp_pinsker_bounds, marker='o', markersize=3, color=r1_colors[idx], label=f"$r_1 = {r1}$")
    axes[1, 1].axhline(y=no_comp_pinsker_bound, color='k', linestyle='--')
    axes[1, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 1].set_xlabel("String length")
    axes[1, 1].set_ylabel("Error bound, Pinsker")
    axes[1, 1].set_ylim(0.85, 1.85)

    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
            ncol=len(r1_vals) + 1, fancybox=True, shadow=True)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plot_path = comp_filepath[:-5] + ".png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.show()


def collect_quant_results(
        dataset_name: str,
        hw: int,
        nl: int,
        comp_scheme: str,
        bound_type: str
):

    no_comp_filepath = get_filepath(dataset_name, hw, nl, "no_comp")
    comp_filepath = get_filepath(dataset_name, hw, nl, comp_scheme)

    no_comp_results = FinalCompResults.load_from_json(filepath=no_comp_filepath).best_results
    all_comp_results = FinalCompResults.load_from_json(filepath=comp_filepath)

    if bound_type == "Inverse KL":
        comp_results, all_equal = all_comp_results.best_inverse_kl_results
        error_bound = comp_results.error_bound_inverse_kl_spectral_domain
    elif bound_type == "Pinsker":
        comp_results, all_equal = all_comp_results.best_pinsker_results
        error_bound = comp_results.error_bound_pinsker_spectral_domain
    else:
        raise ValueError(f"Unknown bound type: {bound_type}. Should be 'Inverse KL' or 'Pinsker'.")

    if comp_scheme == "no_comp":
        comp_values = None
    elif comp_scheme == "quant_k_means":
        comp_values = f"$c={comp_results.codeword_length}$"
    elif comp_scheme == "quant_trunc":
        comp_values = f"$b_e={comp_results.exponent_bits}, b_m={comp_results.mantissa_bits}$"
    elif comp_scheme == "low_rank":
        comp_values = rf"$\bm{{r}}={comp_results.ranks}$"
    elif comp_scheme == "low_rank_and_quant_k_means":
        comp_values = rf"$\bm{{r}}={comp_results.ranks}, c={comp_results.codeword_length}$"
    elif comp_scheme == "low_rank_and_quant_trunc":
        comp_values = rf"$\bm{{r}}={comp_results.ranks}, b_e={comp_results.exponent_bits}, b_m={comp_results.mantissa_bits}$"
    else:
        raise ValueError(f"Unknown compression scheme: {comp_scheme}.")
    
    set_NA = all_equal and comp_scheme != "no_comp"


    summary = {
        "Bound Type": bound_type,
        "Dataset": dataset_name,
        "Hidden Width": hw,
        "Num Hidden Layers": nl,
        "Comp Scheme": comp_schemes[comp_scheme],
        "Comp Values": comp_values,
        "Comp Factor": comp_results.KL / no_comp_results.KL,
        "Margin": comp_results.margin_spectral_domain,
        "Original Train Error": 1 - no_comp_results.train_accuracy,
        "Comp Train Error": 1 - comp_results.train_accuracy,
        "Comp Margin Loss": comp_results.train_margin_loss_spectral_domain,
        "Error Bound": error_bound,
        "Original Test Error": 1 - no_comp_results.test_accuracy,
        "All Equal": all_equal,
    }

    for key in [
        "Comp Values",
        "Comp Factor",
        "Margin",
        "Comp Train Error",
        "Comp Margin Loss",
        "Error Bound",
    ]:
        if set_NA:
            summary[key] = "N/A"

    return summary


if __name__ == "__main__":

    # for hw in [4, 8, 16, 32, 64, 128, 256, 512]:
    for hw in [32]:
        # for nl in [1, 2, 3, 4]:
        for nl in [1]:

            # plot_trunc_results(
            #     dataset_name="MNIST1D",
            #     hw=hw,
            #     nl=nl,
            # )

            # plot_k_means_results(
            #     dataset_name="MNIST1D",
            #     hw=hw,
            #     nl=nl,
            # )

            # if nl == 2:
            plot_low_rank_results(
                dataset_name="MNIST1D",
                hw=hw,
                nl=nl,
            )
            
            # summary = collect_quant_results(
            #     dataset_name="MNIST1D",
            #     hw=hw,
            #     nl=nl,
            #     comp_scheme="quant_trunc",
            #     bound_type="Inverse KL",
            # )
            # print(summary)