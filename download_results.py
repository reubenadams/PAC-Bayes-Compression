import wandb
import pandas as pd


def download_sweep_results_raw(sweep_id, project_name, entity, save_to_path):

    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project_name}/{sweep_id}")
    sweep.runs.per_page = len(sweep.runs)
    runs = sweep.runs

    runs_data = []
    for run in runs:

        # Combine into one row
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            **run.config,
            **run.summary,
        }
        runs_data.append(run_data)
    
    df = pd.DataFrame(runs_data)
    
    df.to_csv(save_to_path, index=False)


def download_sweep_results_clean(sweep_id, project_name, entity, save_to_path, base):
    """
    Download results from a wandb sweep as a pandas DataFrame.
    
    Args:
        sweep_id (str): The ID of the sweep
        project_name (str): The name of your wandb project
        entity (str, optional): Your wandb username or team name
    
    Returns:
        pd.DataFrame: DataFrame containing all runs from the sweep
    """
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project_name}/{sweep_id}")
    runs = sweep.runs
    
    config_ignore = {"input_dim", "output_dim"}
    if base:
        summary_ignore = {"Epoch", "Base Train Loss", "Final Overall Base Train Loss"}
    else:
        summary_ignore = {"Batch Size", "Depth", "Dist Train (kl mean)", "Dropout Probability", "Learning Rate", "Optimizer", "Weight Decay", "Width"}

    runs_data = []
    for run in runs:

        config = {k: v for k, v in run.config.items() if (not k.startswith('_') and not k in config_ignore)}
        summary = {k: v for k, v in run.summary.items() if (not k.startswith('_') and not k in summary_ignore)}

        # Combine into one row
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            **config,
            **summary
        }
        runs_data.append(run_data)
    
    df = pd.DataFrame(runs_data)
    df = df.drop_duplicates(keep="first")
    
    df.to_csv(save_to_path, index=False)


def combine_results(base_path, dist_path, combined_path):
    df_base = pd.read_csv(base_path)
    df_dist = pd.read_csv(dist_path)[["run_name", "Complexity", "Epoch", "Generalization Gap", "KL Loss on Train Data (kl mean)"]]
    df_base.rename(columns={"Epochs Taken": "Epochs Taken Base", "Reached Target": "Reached Target Base"}, inplace=True)
    df_dist.rename(columns={"Epoch": "Epochs Taken Dist"}, inplace=True)
    df_comb = pd.merge(df_base, df_dist, on="run_name", how="left")
    df_comb.to_csv(combined_path, index=False)



if __name__ == "__main__":

    base_results_path_raw = "sweep_results_2187_big_base_raw_new.csv"
    dist_results_path_raw = "sweep_results_2187_big_dist_raw_new.csv"
    base_results_path = "sweep_results_2187_big_base_new.csv"
    dist_results_path = "sweep_results_2187_big_dist_new.csv"
    comb_results_path = "sweep_results_2187_big_comb_new.csv"
    download_sweep_results_raw(
        sweep_id="7spkiovz",
        project_name="2187-big",
        entity="teamreuben",
        save_to_path=base_results_path_raw
        )
    download_sweep_results_raw(
        sweep_id="e8qaxkrj",
        project_name="2187-big",
        entity="teamreuben",
        save_to_path=dist_results_path_raw
        )
    download_sweep_results_clean(
        sweep_id="7spkiovz", 
        project_name="2187-big",
        entity="teamreuben",
        save_to_path=base_results_path,
        base=True
    )
    download_sweep_results_clean(
        sweep_id="e8qaxkrj", 
        project_name="2187-big",
        entity="teamreuben",
        save_to_path=dist_results_path,
        base=False
    )
    combine_results(
        base_results_path,
        dist_results_path,
        comb_results_path,
    )