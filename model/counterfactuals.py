import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import os

def counterfactuals(model, X_train, X_test, actionable_features, target_value, num_solutions, reduced_features_path, user_input=None):
    if X_test is None:
        X_test = X_train.sample(n=min(100, len(X_train)), random_state=42)

    # Load the reduced feature list
    features_txt = 'reduced_training_data_features.txt'
    if not os.path.exists(features_txt):
        print(f"File {features_txt} does not exist.")
        return [], {}, pd.DataFrame()

    with open(features_txt, 'r') as f:
        reduced_features = [line.strip() for line in f.readlines()]

    X_train = X_train[reduced_features]

    if user_input is not None:
        instance = pd.DataFrame(user_input, index=[0])
    else:
        instance = X_test.iloc[0:1]
    instance = instance[reduced_features]

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)

    # Get SHAP importance values for each feature
    shap_importance = {
        feature: np.abs(shap_values[0][X_train.columns.get_loc(feature)])
        for feature in X_train.columns
    }
    shap_importance_df = pd.DataFrame(list(shap_importance.items()), columns=['feature', 'importance'])
    shap_importance_df = shap_importance_df.sort_values(by='importance', ascending=False)

    counterfactuals_list = []
    actionable_impacts = {feature: 0 for feature in actionable_features}

    #SHAP for actionable features
    shap_actionable_values = {
        feature: shap_values[0][X_train.columns.get_loc(feature)] for feature in actionable_features
    }

    original_prediction = model.predict(instance)[0]

    for _ in range(num_solutions):
        modified_instance = instance.copy()

        #only actionable features
        for feature in actionable_features:
            if feature in modified_instance.columns:
                # Get SHAP value for the feature
                impact = shap_values[0][X_train.columns.get_loc(feature)]
                modified_instance[feature] += impact * 0.01 #modify based on impact

        # Calculate new prediction
        counterfactual_prediction = model.predict(modified_instance)[0]

        # If prediction is close  to target_value, accept it
        if np.abs(counterfactual_prediction - target_value) < 1.0:
            counterfactuals_list.append(modified_instance)

            # Log how the actionable features were changed based on SHAP values
            for feature in actionable_features:
                if feature in modified_instance.columns:
                    original_value = instance[feature].values[0]
                    modified_value = modified_instance[feature].values[0]
                    actionable_impacts[feature] += np.abs(original_value - modified_value)

    # Normalize the actionable impacts if counterfactuals were found
    if counterfactuals_list:
        for feature in actionable_impacts:
            actionable_impacts[feature] /= len(counterfactuals_list)

    return counterfactuals_list, actionable_impacts, shap_importance_df, shap_actionable_values

import xgboost as xgb
import matplotlib.pyplot as plt
import graphviz

def plot_counterfactuals(counterfactuals_list, model, X_train, feature_names, target_value):
    """
    Visualizes feature changes in the context of a decision tree.
    Plots a decision tree with counterfactuals shown as different colors or annotations.

    Parameters:
    - counterfactuals_list: List of counterfactuals generated to reach the target value.
    - model: The trained XGBoost model.
    - X_train: The training data used to train the model.
    - feature_names: List of feature names in the model.
    - target_value: The target prediction value we want to achieve.
    """
    # Plot tree
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_tree(model, num_trees=0, ax=ax)

    # Annotate or highlight counterfactuals in the tree
    for i, cf in enumerate(counterfactuals_list):
        # Predict the counterfactuals to see which class they fall into
        cf_prediction = model.predict(cf)[0]
        
        # Get the index of the leaf node the counterfactual reaches
        leaf_index = model.apply(cf)[0]

        # Annotate the leaf node
        ax.text(0.5, 0.5, f"Counterfactual {i+1}\nPredicted: {cf_prediction}", 
                color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5), ha='center')

    # Adjust the layout and display
    plt.tight_layout()
    plt.show()
