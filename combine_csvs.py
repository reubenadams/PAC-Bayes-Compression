import pandas as pd
import glob
import os


def combine_csv_files(directory_path: str) -> None:
    """Combine all CSV files in the specified directory into a single CSV file named 'combined.csv',
    excluding any existing 'combined.csv' file."""
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Get all CSV files
    all_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    # Exclude combined.csv from the list if it exists
    output_path = os.path.join(directory_path, "combined.csv")
    all_files = [file for file in all_files if os.path.basename(file) != "combined.csv"]
    
    if not all_files:
        raise ValueError(f"No CSV files found in directory: {directory_path}")
    
    df_list = []
    
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # This will overwrite combined.csv if it exists
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined {len(all_files)} CSV files into {output_path}")


if __name__ == "__main__":
    combine_csv_files(r"distillation\models\MNIST1D\40\dist_metrics")