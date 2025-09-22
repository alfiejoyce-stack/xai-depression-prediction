import os
import pandas as pd
from glob import glob

# Define the directory containing the .xpt files
directory = r"C:\Users\wsaj5\OneDrive - Loughborough University\Individual Project\Python\Datasets IP\Datasets All"

# Get a list of all .xpt files in the directory
xpt_files = glob(os.path.join(directory, "*.xpt"))

# Initialize an empty dataframe
empty_dataframe = None

# Loop through each .xpt file and merge on SEQN
for file in xpt_files:
    try:
        # Read the .xpt file
        df = pd.read_sas(file, format="xport", encoding="utf-8")

        # Ensure SEQN is present
        if "SEQN" not in df.columns:
            print(f"Skipping {file} (no SEQN column)")
            continue

        # Merge datasets using an OUTER JOIN to keep all SEQN values
        if empty_dataframe is None:
            empty_dataframe = df
        else:
            # Merge the dataframes based on SEQN
            empty_dataframe = pd.merge(empty_dataframe, df, on="SEQN", how="outer", suffixes=("", "_2"))

            # For each column in the new dataframe, combine missing values
            for col in df.columns:
                if col != "SEQN" and col + "_2" in empty_dataframe.columns:
                    empty_dataframe[col] = empty_dataframe[col].combine_first(empty_dataframe[col + "_2"])
                    empty_dataframe.drop(columns=[col + "_2"], inplace=True)

        print(f"Successfully merged: {file}")
        print(f"Number of rows after merging {file}: {len(empty_dataframe)}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Check if the merged dataframe has data
if empty_dataframe is not None:
    # Fill missing values in numeric columns with the median, if any numeric columns exist
    numeric = empty_dataframe.select_dtypes(include=['number']).columns
    if len(numeric) > 0:
        empty_dataframe[numeric] = empty_dataframe[numeric].fillna(empty_dataframe[numeric].median())

    # Fill missing values in non-numeric columns (like categorical ones) with 'Unknown'
    non_numeric = empty_dataframe.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        empty_dataframe[non_numeric] = empty_dataframe[non_numeric].fillna('Unknown')

    # Save the final merged dataset
    output_file = os.path.join(directory, "merged_all_seqn.csv")
    empty_dataframe.to_csv(output_file, index=False)

    print(f"\nFinal dataset saved to: {output_file}")
else:
    print("No valid data to merge.")
