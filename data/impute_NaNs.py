import pandas as pd
from sklearn.impute import KNNImputer

# Step 1: Load the CSV file
df = pd.read_csv(r'C:\Users\wsaj5\Documents\ML\training_data.csv')

# Step 2: Extract features and target variable
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values   # Target (the last column)

# Step 3: Apply KNN Imputer to fill missing values in the target (y)
imputer = KNNImputer(n_neighbors=5)  # Using 5 nearest neighbors
y_imputed = imputer.fit_transform(y.reshape(-1, 1)).ravel()

# Step 4: Replace NaNs in y with the imputed values
df[df.columns[-1]] = y_imputed

# Step 5: Save the fixed data to a new CSV file
df.to_csv(r'C:\Users\wsaj5\Documents\ML\training_data_fixed.csv', index=False)

print("NaN values in target variable have been imputed and saved to 'training_data_fixed.csv'")
