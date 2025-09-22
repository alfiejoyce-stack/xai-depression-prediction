import pandas as pd

# Path to the merged dataset
input_file = r"C:\Users\wsaj5\OneDrive - Loughborough University\Individual Project\Python\Datasets IP\Datasets All\merged_all_seqn.csv"

# Load the merged dataset
merged_df = pd.read_csv(input_file, low_memory=False)


# Remove duplicate rows based on the SEQN column, keeping the first occurrence
merged_df_cleaned = merged_df.drop_duplicates(subset="SEQN", keep="first")

# Path to save the cleaned dataset
output_file_cleaned = r"C:\Users\wsaj5\Downloads\Datasets All\merged_all_seqn_no_duplicates.csv"

# Save the cleaned dataframe to a new file
merged_df_cleaned.to_csv(output_file_cleaned, index=False)

print(f"Final dataset with duplicates removed saved to: {output_file_cleaned}")
