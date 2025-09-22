import pandas as pd

# Define file path for the reduced features
file_path = 'reduced_features.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Get the number of rows and columns
num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Calculate the correlation matrix
correlation_matrix = df.corr().abs()  # Absolute correlation values

# Set a correlation threshold (e.g., 0.9)
correlation_threshold = 0.7

# Identify features to drop
to_drop = set()
removed_features = []  # List to store removed features

# Loop over the correlation matrix to find pairs with correlation above the threshold
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > correlation_threshold:
            # If correlation is higher than threshold, drop one feature from the pair
            colname = correlation_matrix.columns[i]
            if colname not in to_drop:
                to_drop.add(colname)
                removed_features.append(colname)

# Drop the identified highly correlated features
df_reduced = df.drop(columns=to_drop)

# Save the reduced dataset to a new CSV file
df_reduced.to_csv('reduced_features_after_correlation_reduction.csv', index=False)

# Print the removed features due to correlation
print(f"Removed features due to correlation above {correlation_threshold}:")
print(removed_features)

print(f'Reduced features saved to "reduced_features_after_correlation_reduction.csv" with {df_reduced.shape[1]} features.')
