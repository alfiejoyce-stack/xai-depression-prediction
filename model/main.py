import warnings
import data_loader
import modeling
from counterfactuals import counterfactuals, plot_counterfactuals
from modeling import preprocess_data
from user_input_gui import get_user_input_gui
from project_phq9_and_suggest import project_phq9_and_suggest
from dashboard_visuals import plot_dashboard
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from heatmap import plot_feature_importance_and_correlation

warnings.filterwarnings('ignore')

# Define file path
file_path = 'training_data.csv'
reduced_features_path = 'reduced_features.csv'

# Loads the datasets
X, y, reduced_features = data_loader.data_loader(file_path, reduced_features_path)

# Preprocess the data (without including target variable 'y' in X)
X_preprocessed, y_processed = preprocess_data(X, y)  

# Train the model using the preprocessed data
model, X_test, y_test, X_train, y_train = modeling.train_model(X_preprocessed, y_processed)

# After training the model, visualize feature importance and correlations
plot_feature_importance_and_correlation(model, X_train)

# Counterfactuals
num_solutions = 5

# Feature Name Mapping 
feature_name_map = {
    'KIQ481': 'Nighttime urination frequency',
    'LBXBPB': 'Blood lead level (µg/dL)',
    'RIDAGEYR': 'Age (years)',
    'LBXSHBG': 'Sex hormone-binding globulin (nmol/L)',
    'LBXMOPCT': 'Monocyte percentage (%)',
    'ALQ121': 'Alcohol use frequency (past 12 months)',
    'FNDAEDI': 'Disability indicator (WG-SS Enhanced)',
    'ALQ142': 'Alcohol frequency: ≥4 drinks (past year)',  
    'LBXAND': 'Androstenedione (ng/dL)',
    'DR1TFA': 'Folic acid (mcg)',
    'PHAFSTMN': 'Total length of food fast (minutes)',
    'LBXPLTSI': 'Platelet count (1000 cells/uL)',
    'LBDNENO': 'Segmented neutrophils num (1000 cell/uL)',
    'OHQ680': 'Last year embarrassed because of mouth',
    'KIQ052': 'During the past 12 months, how much did your leakage of urine affect your day-to-day activities?',
    'HUQ090': 'Seen mental health professional/past year?',
    'SMDANY': 'Used any tobacco product last 5 days?',
    'HSQ590': 'Blood ever tested for HIV virus?',
    'RIDEXAGM': 'Age in months at exam - 0 to 19 years',
    'LBX17H': '17α-hydroxyprogesterone (ng/dL)',
}

# Updated list of actionable features
actionable_features = [
    'ALQ121',  # Alcohol use frequency (past 12 months)
    'ALQ142',  # Alcohol frequency: ≥4 drinks (past year)
    'SMDANY',  # Used any tobacco product last 5 days
    'PHAFSTMN', # Total length of food fast (minutes) - can be affected by dietary habits
    'DR1TFA',  # Folic acid (mcg) - dietary change, supplementation
    'OHQ680',  # Last year embarrassed because of mouth - related to dental hygiene 
]

# Get the user input from the GUI
user_data_df, date_series, feature_order = get_user_input_gui("reduced_training_data.csv", feature_name_map)
print("DEBUG - DataFrame Columns:", user_data_df.columns.tolist())

if user_data_df is not None:
    # Select the latest input 
    latest_input_row = user_data_df.iloc[-1]

    # Ensure that date column is included and properly formatted 
    latest_input_row['date'] = pd.to_datetime(latest_input_row['date'], errors='coerce')
    
    # Project and guide the user based on their latest input
    future_phq9, target_value, counterfactuals_result, user_data_df = project_phq9_and_suggest(
        user_data_df=user_data_df,
        date_series=date_series,
        model=model,
        X_train=X_train,
        X_test=X_test,
        actionable_features=actionable_features,
        reduced_features_path=reduced_features_path,
        counterfactuals_func=counterfactuals,
        num_solutions=num_solutions,
        latest_input_row=latest_input_row
    )

    print("Final collected user data:")
    print(user_data_df.head())

    print("Dates:")
    print(date_series)
    
    # Extract counterfactuals from the result
    counterfactuals_list = counterfactuals_result
    counterfactual_summary = f"{len(counterfactuals_list)} counterfactual(s) found."

    # Print summary of changes for the counterfactuals
    print("\n Counterfactuals for the Latest User Input")
    print(counterfactual_summary)
    
    # Display the counterfactual solutions for the latest input
    print(counterfactuals_list)


# Counterfactuals and SHAP feature importance
counterfactuals_list, _, shap_importance_df, _ = counterfactuals(
    model=model,
    X_train=X_train,
    X_test=X_test,
    actionable_features=actionable_features,
    target_value=target_value,
    num_solutions=num_solutions,
    reduced_features_path=reduced_features_path,
    user_input=user_data_df.iloc[-1].to_dict()
)

# Decision tree
plot_counterfactuals(
    counterfactuals_result,
    model=model,
    X_train=X_train,
    feature_names=feature_name_map,
    target_value=target_value
)

# Process SHAP importance for actionable features
if 'feature' not in shap_importance_df.columns:
    raise ValueError("SHAP importance DataFrame is missing the 'feature' column.")

# Set of SHAP features
shap_features = set(shap_importance_df['feature'])
actionable_set = set(actionable_features)

# Which actionable features are present in SHAP DataFrame
present_actionable = list(actionable_set & shap_features)
missing_actionable = list(actionable_set - shap_features)

# Log missing ones
if missing_actionable:
    print("Warning: The following actionable features are missing from SHAP importance dataframe:")
    for feat in missing_actionable:
        print(f"  - {feature_name_map.get(feat, feat)} ({feat})")

# Extract importance values for those present
if present_actionable:
    shap_df_indexed = shap_importance_df.set_index('feature')
    actionable_impacts = shap_df_indexed.loc[present_actionable]['importance'].to_dict()
    print("\n Actionable SHAP impacts:")
    for f, v in actionable_impacts.items():
        print(f"  - {feature_name_map.get(f, f)}: {v:.4f}")
else:
    print(" Error: No actionable features found in SHAP importance dataframe.")
    actionable_impacts = {}

plot_dashboard(
    user_df=user_data_df,
    shap_actionable_values=actionable_impacts,
    shap_importance_df=shap_importance_df,
    feature_name_map=feature_name_map
)
