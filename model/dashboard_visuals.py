import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

sns.set(style="whitegrid")

def plot_dashboard(user_df,
                   shap_actionable_values,  
                   shap_importance_df,
                   feature_name_map):
    """
    Generates a dashboard with visualizations for SHAP feature importance and
    the SHAP values of actionable features (positive and negative impact).
    
    Parameters:
    - user_df: original user input DataFrame (with raw feature names)
    - shap_actionable_values: dict of {feature_name: shap_value}
    - shap_importance_df: DataFrame of overall SHAP importance
    - feature_name_map: mapping from raw feature names to readable labels
    """

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2)

    #Top row
    ax1 = fig.add_subplot(gs[0, :])  
    if shap_importance_df.empty:
        ax1.text(0.5, 0.5, 'No SHAP importance data available', ha='center', va='center', color='gray')
        ax1.axis('off')
    else:
        sns.barplot(x='importance', y='feature', data=shap_importance_df, ax=ax1, palette='Blues_d')
        ax1.set_title('SHAP Feature Importance')
        ax1.set_xlabel('SHAP Importance')
        ax1.set_ylabel('Feature')

    #Middle row
    ax2 = fig.add_subplot(gs[1, :])
    
    if not shap_actionable_values:
        ax2.text(0.5, 0.5, 'No actionable SHAP values available', ha='center', va='center', color='gray')
        ax2.axis('off')
    else:
        items = [(feature_name_map.get(k, k), v) for k, v in shap_actionable_values.items()]
        items = sorted(items, key=lambda x: abs(x[1]), reverse=True)
        
        labels, values = zip(*items)
        colors = ['lightcoral' if val > 0 else 'palegreen' for val in values]
        
        sns.barplot(x=values, y=labels, palette=colors, ax=ax2, orient='h')
        ax2.set_title('Actionable Feature SHAP Impact on PHQ-9')
        ax2.set_xlabel('SHAP Value (Impact on Prediction)')

        increase_patch = mpatches.Patch(color='lightcoral', label='Increases PHQ-9')
        decrease_patch = mpatches.Patch(color='palegreen', label='Decreases PHQ-9')
        ax2.legend(handles=[increase_patch, decrease_patch], loc='best')

    plt.tight_layout()
    plt.savefig("dashboard_output.png", format='png')
    plt.show()
