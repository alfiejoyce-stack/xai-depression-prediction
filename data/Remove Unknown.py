import numpy as np
import pandas as pd

# Define the input file path for the file with duplicates removed
input_file = r"C:\Users\wsaj5\OneDrive - Loughborough University\Individual Project\Python\Datasets IP\Datasets All\merged_all_seqn_no_duplicates.csv"

# Load the dataset
merged_df = pd.read_csv(input_file, low_memory=False)

# Replace 'Unknown' with NaN in the entire dataframe (case-insensitive)
merged_df = merged_df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip().lower() == 'unknown' else x)

# Fill missing values in numeric columns with median
numeric_cols = merged_df.select_dtypes(include=['number']).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

# Fill missing values in non-numeric columns with 'Unknown'
non_numeric_cols = merged_df.select_dtypes(exclude=['number']).columns
merged_df[non_numeric_cols] = merged_df[non_numeric_cols].fillna('Unknown')

# Save the cleaned dataframe to a new file
output_file = r"C:\Users\wsaj5\OneDrive - Loughborough University\Individual Project\Python\Datasets IP\Datasets All\merged_all_seqn_no_unknowns.csv"
merged_df.to_csv(output_file, index=False)

print(f"\nFinal dataset with 'Unknown' replaced saved to: {output_file}")
