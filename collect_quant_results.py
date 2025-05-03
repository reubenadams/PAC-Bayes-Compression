import os
from config import FinalCompResults
from pprint import pprint
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


# nums = [0, 0.15, 0.29, 0.42, 0.54, 0.65, 0.75, 0.84, 0.92, 1]
# colors = plt.cm.viridis(nums)

colors = plt.cm.viridis(np.linspace(0, 1, 9))


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

    no_comp_results = FinalCompResults.load_from_json(filepath=no_comp_filepath).best_results
    all_comp_results = FinalCompResults.load_from_json(filepath=comp_filepath).all_results

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
        axes[0, 0].plot(comp_string_lengths, comp_train_errors, color=colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[0, 0].axhline(y=no_comp_error, color='k', linestyle='--', label="No Compression")
    axes[0, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 0].set_ylabel("Error on train set")
    axes[0, 0].set_ylim(0.25, 1.05)

    # Plot 2: Margin Losses (top-right)
    for exponent_bits in exponent_range:
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
        comp_margin_losses = [results.train_margin_loss_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
        axes[0, 1].plot(comp_string_lengths, comp_margin_losses, color=colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[0, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[0, 1].set_ylabel("Margin loss on train set")
    axes[0, 1].set_ylim(0.25, 1.05)

    # Plot 3: Inverse KL Bounds (bottom-left)
    for exponent_bits in exponent_range:
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
        comp_inverse_kl_bounds = [results.error_bound_inverse_kl_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
        axes[1, 0].plot(comp_string_lengths, comp_inverse_kl_bounds, color=colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[1, 0].axhline(y=no_comp_inverse_kl_bound, color='k', linestyle='--')
    axes[1, 0].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 0].set_xlabel("String length")
    axes[1, 0].set_ylabel("Error bound, inverse kl")

    # Plot 4: Pinsker Bounds (bottom-right)
    for exponent_bits in exponent_range:
        comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
        comp_pinsker_bounds = [results.error_bound_pinsker_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
        axes[1, 1].plot(comp_string_lengths, comp_pinsker_bounds, color=colors[exponent_bits], label=f"$b_e = {exponent_bits}$")
    axes[1, 1].axhline(y=no_comp_pinsker_bound, color='k', linestyle='--')
    axes[1, 1].axvline(x=no_comp_string_length, color='k', linestyle='--')
    axes[1, 1].set_xlabel("String length")
    axes[1, 1].set_ylabel("Error bound, Pinsker")

    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
            ncol=5, fancybox=True, shadow=True)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plot_path = comp_filepath[:-5] + ".png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


    # Individual plots
    # # Plot comp train errors
    # for exponent_bits in range(3, 9):
    #     comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     comp_train_errors = [1 - results.train_accuracy for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     plt.plot(comp_string_lengths, comp_train_errors, color=colors[exponent_bits], label=f"{exponent_bits} Exponent Bits")
    # plt.axhline(y=no_comp_error, color='k', linestyle='--', label="No Compression")
    # plt.axvline(x=no_comp_string_length, color='k', linestyle='--')
    # plt.xlabel("String Length |s|")
    # plt.ylabel("Train Error")
    # plt.legend()
    # plt.show()

    # # Plot comp margin losses
    # for exponent_bits in range(3, 9):
    #     comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     comp_margin_losses = [results.train_margin_loss_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     plt.plot(comp_string_lengths, comp_margin_losses, color=colors[exponent_bits], label=f"{exponent_bits} Exponent Bits")
    # plt.axvline(x=no_comp_string_length, color='k', linestyle='--')
    # plt.xlabel("String Length |s|")
    # plt.ylabel("Train Margin Loss")
    # plt.legend()
    # plt.show()

    # # Plot comp inverse KL bounds
    # for exponent_bits in range(3, 9):
    #     comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     comp_inverse_kl_bounds = [results.error_bound_inverse_kl_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     plt.plot(comp_string_lengths, comp_inverse_kl_bounds, color=colors[exponent_bits], label=f"{exponent_bits} Exponent Bits")
    # plt.axhline(y=no_comp_inverse_kl_bound, color='k', linestyle='--', label="No Compression")
    # plt.axvline(x=no_comp_string_length, color='k', linestyle='--')
    # plt.xlabel("String Length |s|")
    # plt.ylabel("Inverse KL Bound")
    # plt.legend()
    # plt.show()

    # # Plot comp Pinsker bounds
    # for exponent_bits in range(3, 9):
    #     comp_string_lengths = [results.KL / sqrt(2) for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     comp_pinsker_bounds = [results.error_bound_pinsker_spectral_domain for results in all_comp_results if results.exponent_bits == exponent_bits]
    #     plt.plot(comp_string_lengths, comp_pinsker_bounds, color=colors[exponent_bits], label=f"{exponent_bits} Exponent Bits")
    # plt.axhline(y=no_comp_pinsker_bound, color='k', linestyle='--', label="No Compression")
    # plt.axvline(x=no_comp_string_length, color='k', linestyle='--')
    # plt.xlabel("String Length |s|")
    # plt.ylabel("Pinsker Bound")
    # plt.legend()
    # plt.show()



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

    # for hw in [32]:
    #     for nl in [1, 2]:
    #         print(f"HW: {hw}, NL: {nl}")
    #         try:
    #             for comp_scheme in comp_schemes:
    #                 summary = collect_quant_results(
    #                     dataset_name="MNIST1D",
    #                     hw=hw,
    #                     nl=nl,
    #                     comp_scheme=comp_scheme,
    #                     # bound_type="Inverse KL"
    #                     bound_type="Pinsker",
    #                 )
    #                 pprint(summary)
    #                 print()
    #             print()
    #             print()
    #         except FileNotFoundError as e:
    #             print(f"File not found: {e}")
    #             print()
    #             print()
    #             continue

    for hw in [4, 8, 16, 32, 64, 128, 256, 512]:
        for nl in [1, 2, 3, 4]:
            plot_trunc_results(
                dataset_name="MNIST1D",
                hw=hw,
                nl=nl,
            )
