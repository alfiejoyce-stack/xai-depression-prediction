import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Define file path
file_path = 'training_data.csv'

# Load dataset
df = pd.read_csv(file_path)

# Get the number of rows and columns
num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Ensure 'SEQN' is present and separate it
if 'SEQN' in df.columns:
    seqn = df[['SEQN']]  # Store SEQN separately
    X = df.drop(columns=['SEQN'])  # Remove SEQN from features
else:
    seqn = None
    X = df.iloc[:, :-1]  # Features (excluding the last column)

y = df.iloc[:, -1]  # Target variable (last column)

# Encode categorical features (if any) in X
label_encoder = LabelEncoder()

for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

# Train Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
regressor.fit(X, y)

# Get feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': regressor.feature_importances_
})

# Sort features by importance and select the top n
top_features = feature_importance_df.sort_values(by='Importance', ascending=False).iloc[:50]['Feature'].tolist()

# Keep only the top n features
X_reduced = X[top_features]

# Reinsert SEQN at the front if it exists
if seqn is not None:
    X_reduced = pd.concat([seqn, X_reduced], axis=1)

# Save the reduced dataset to a new CSV file
X_reduced.to_csv('reduced_features.csv', index=False)

print(f'Reduced features saved to "reduced_features.csv" with {X_reduced.shape[1]} features (including SEQN).')
