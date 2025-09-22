import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the reduced dataset
reduced_data_path = 'reduced_features.csv'
X_reduced = pd.read_csv(reduced_data_path)

# Drop the 'SEQN' column if it exists (it should not be in the correlation matrix)
if 'SEQN' in X_reduced.columns:
    X_reduced = X_reduced.drop(columns=['SEQN'])

# Calculate the correlation matrix
correlation_matrix = X_reduced.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Set plot title
plt.title('Feature Dependence Heatmap')

# Show the plot
plt.show()
