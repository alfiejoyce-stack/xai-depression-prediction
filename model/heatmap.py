import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance_bar(model, X_train, top_n=20):
    """
    Plot a bar chart for the top n important features using model's feature importances.
    
    Parameters:
    - model: Trained machine learning model (e.g., RandomForest, XGBRegressor).
    - X_train: The feature set used for training.
    - top_n: Number of top important features to visualize.
    
    
    - Displays bar plot.
    """
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values(by='importance', ascending=False).head(top_n)

    # Create a bar plot for the most important features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(X, threshold=0.9):
    """
    Plot a heatmap for the correlation matrix of features.
    
    Parameters:
    - X: DataFrame containing features.
    - threshold: Correlation threshold to focus on.
    
    
    - Displays heatmap.
    """
    # Correlation matrix
    corr_matrix = X.corr().abs()

    # Upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt=".2f", cbar_kws={'shrink': .8})
    plt.title("Feature Correlation Matrix")
    plt.show()

def plot_feature_importance_and_correlation(model, X_train, top_n=20):
    """
    Create a figure with two subplots:
    - Feature Importance bar plot
    - Correlation heatmap
    
    Parameters:
    - model: Trained model (e.g., RandomForest, XGBRegressor).
    - X_train: Training data features.
    - top_n: Number of important features to show.
    
    - Displays the plots.
    """
    # Two plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Feature importance bar plot
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values(by='importance', ascending=False).head(top_n)

    sns.barplot(data=importance, x='importance', y='feature', palette='viridis', ax=axes[0])
    axes[0].set_title(f"Top {top_n} Feature Importances")
    axes[0].set_xlabel("Importance")
    axes[0].set_ylabel("Feature")

    # Correlation heatmap
    corr_matrix = X_train.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt=".2f", cbar_kws={'shrink': .8}, ax=axes[1])
    axes[1].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.show()
