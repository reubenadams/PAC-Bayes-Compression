import wandb
import pandas as pd

def download_sweep_results(sweep_id, project_name, entity):
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
    
    # Get the sweep object
    sweep = api.sweep(f"{entity}/{project_name}/{sweep_id}")
    
    # Get all runs in the sweep
    runs = sweep.runs
    
    # Extract the data we want
    runs_data = []
    for run in runs:
        # Get config (hyperparameters)
        config = {f"config/{k}": v for k, v in run.config.items() 
                 if not k.startswith('_')}
        
        # Get summary (metrics)
        summary = {f"metric/{k}": v for k, v in run.summary.items() 
                  if not k.startswith('_')}
        
        # Combine into one row
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            **config,
            **summary
        }
        runs_data.append(run_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(runs_data)
    
    return df

# Example usage
sweep_results = download_sweep_results(
    sweep_id="avg612py", 
    project_name="base-more-epochs",
    entity="teamreuben"
)

# Save to CSV
sweep_results.to_csv("sweep_results_more_epochs.csv", index=False)