import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def data_loader(file_path, reduced_features_path):
    # Load the main dataset
    df = pd.read_csv(file_path)

    # Load reduced and actionable features (not used in this function directly, but loaded)
    reduced_features = pd.read_csv(reduced_features_path)

    # Extract target variable (last column)
    y = df.iloc[:, -1]  # Keep as pandas Series

    # Replace NaNs in y with the median
    y.fillna(y.median(), inplace=True)

    # Extract features (excluding the target column)
    X = df.iloc[:, :-1]  # Keep as pandas DataFrame

    # Separate categorical & numerical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    # One-Hot Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    x_categorical = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    x_categorical.columns = encoder.get_feature_names_out(categorical_cols)

    # Reconstruct feature DataFrame
    X = pd.concat([X[numerical_cols].reset_index(drop=True), x_categorical.reset_index(drop=True)], axis=1)

    return X, y, reduced_features
