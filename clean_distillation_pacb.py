import pandas as pd
import wandb


def get_failed_run_names():

    pac_cols = ["max_sigma", "noisy_error", "noise_trials", "pac_bound_inverse_kl" ,"pac_bound_pinsker"]

    df = pd.read_csv('sweep_results_2187_big_comb.csv')
    for col_name in pac_cols:
        if col_name not in df.columns:
            return []
    df_reached_target = df[df['Reached Target Base'] == True]
    df_missing_pac = df_reached_target[df_reached_target[pac_cols].isna().any(axis=1)]
    failed_run_names = df_missing_pac["run_name"].values
    return failed_run_names


def delete_failed_runs(failed_run_names):
    api = wandb.Api()

    # Replace with your project and sweep ID
    project_name = "2187-big"
    sweep_id = "rm2313rt"

    # Get all runs in the sweep
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    runs = sweep.runs

    for run in runs:
        if run.name in failed_run_names:
            print(f"Deleting run: {run.name} (ID: {run.id})")
            run.delete()


failed_run_names = get_failed_run_names()
print(failed_run_names)
# delete_failed_runs(failed_run_names)