import pandas as pd

def analyze_hyperparameter_success(csv_path, hyperparameter_name, hyperparameter_value):
    """
    Analyzes a CSV file of wandb sweep results to determine the success rate of runs
    with a specific hyperparameter value.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing wandb sweep results
    hyperparameter_name : str
        Name of the hyperparameter to analyze (column name in the CSV)
    hyperparameter_value : any
        The specific value of the hyperparameter to filter for
        
    Returns:
    --------
    float
        Proportion of runs with the specified hyperparameter value that reached the target
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if the hyperparameter exists in the dataframe
    if hyperparameter_name not in df.columns:
        raise ValueError(f"Hyperparameter '{hyperparameter_name}' not found in the CSV file")
    
    # Get the name of the target column (assumed to be the last column)
    target_column = df.columns[-1]
    
    # Filter rows with the specific hyperparameter value
    filtered_df = df[df[hyperparameter_name] == hyperparameter_value]
    
    # Calculate the proportion of successful runs
    if len(filtered_df) == 0:
        return 0.0  # No runs with this hyperparameter value
    
    success_rate = filtered_df[target_column].mean()
    
    # Return additional information
    print(f"Total runs with {hyperparameter_name}={hyperparameter_value}: {len(filtered_df)}")
    print(f"Successful runs: {filtered_df[target_column].sum()}")
    print(f"Failed runs: {len(filtered_df) - filtered_df[target_column].sum()}")
    
    return success_rate


success_rate = analyze_hyperparameter_success("sweep_results_more_epochs.csv", "config/dropout_prob", 0.2)
print(f"{success_rate=}")