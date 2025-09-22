import pandas as pd

# Define file path
path = r'C:\Users\wsaj5\Documents\ML\training_data.csv'
output = r'C:\Users\wsaj5\Documents\ML\training_data_phq9.csv'

# Load dataset
dataframe = pd.read_csv(path)

# List of DPQ columns to sum
dpq = ["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"]

# Ensure the columns exist in the dataset
columns = [col for col in dpq if col in dataframe.columns]

# Convert columns to numeric (in case of string values)
dataframe[columns] = dataframe[columns].apply(pd.to_numeric, errors='coerce')

# Filter rows
rows = dataframe[columns].apply(lambda row: row.isin([0, 1, 2, 3]).all(), axis=1)

# Compute PHQ-9 score for valid rows
dataframe.loc[rows, "PHQ9_Score"] = dataframe.loc[rows, columns].sum(axis=1)

# Save the updated DataFrame
dataframe.to_csv(output, index=False)

