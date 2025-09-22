from counterfactuals import counterfactuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def project_phq9_and_suggest(user_data_df, date_series, model, X_train, X_test, actionable_features, reduced_features_path, counterfactuals_func, num_solutions, latest_input_row, return_fig=False):

    if return_fig:
        fig = plt.figure(figsize=(12, 6))
    else:
        plt.figure(figsize=(12, 6))

    user_data_df['date'] = pd.to_datetime(date_series.reset_index(drop=True), errors='coerce')

    if user_data_df['date'].isna().any():
        print("Dropping rows with invalid dates...")
        user_data_df = user_data_df.dropna(subset=['date'])

    if user_data_df.empty:
        raise ValueError("No valid data left after date cleaning")

    # Calculate the days since start
    user_data_df['days_since_start'] = (user_data_df['date'] - user_data_df['date'].min()).dt.days

    # Sort by date
    user_data_df = user_data_df.sort_values('date').reset_index(drop=True)

    # Filter features
    columns_to_remove = ['date', 'PHQ9_Predicted', 'PHQ-9_Predicted', 'phq9_fit', 'days_since_start']
    feature_columns = [col for col in user_data_df.columns if col not in columns_to_remove]

    # Convert to numeric
    X_user = user_data_df[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    if X_user.isnull().any().any():
        print("Found NaNs in input data. Dropping rows with missing values...")
        valid_idx = ~X_user.isnull().any(axis=1)
        user_data_df = user_data_df.loc[valid_idx].reset_index(drop=True)
        X_user = X_user.loc[valid_idx].reset_index(drop=True)

    if user_data_df.empty:
        raise ValueError("No valid rows left after dropping NaNs")

    # Predict PHQ-9 using the model
    print("X_user shape:", X_user.shape)
    phq9_preds = model.predict(X_user)
    user_data_df['PHQ9_Predicted'] = phq9_preds

    # Trendline regression (days since start)
    X_days = user_data_df[['days_since_start']]
    y_phq9 = user_data_df['PHQ9_Predicted']

    reg = LinearRegression()
    reg.fit(X_days, y_phq9)
    user_data_df['phq9_fit'] = reg.predict(X_days)

    last_date = user_data_df['date'].max()
    target_date = last_date + pd.DateOffset(months=1)
    future_day = (target_date - user_data_df['date'].min()).days

    # Predict using regression line
    future_phq9 = reg.predict(np.array([[future_day]]))[0]
    target_value = future_phq9 - 0.5


    print(f"\nProjected PHQ-9 on {target_date.date()}: {future_phq9:.2f}")
    print(f"Target PHQ-9 Score (wellness goal): {target_value:.2f}")
    if return_fig:
        print(f"\nDidn't plot")
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(user_data_df['date'], user_data_df['PHQ9_Predicted'], 'o-', label='Predicted PHQ-9')
        plt.plot(user_data_df['date'], user_data_df['phq9_fit'], 'r-', label='Line of Best Fit')
        plt.scatter(target_date, future_phq9, color='purple', s=100, label='Projected (1 Month)', zorder=5)
        plt.axhline(target_value, color='green', linestyle='--', label=f'Target PHQ-9 = {target_value:.2f}')
        plt.title('PHQ-9 Predictions Over Time')
        plt.xlabel('Date')
        plt.ylabel('Predicted PHQ-9')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        latest_instance = user_data_df.drop(columns=[col for col in columns_to_remove if col in user_data_df.columns]).iloc[[-1]]

        counterfactuals_list, actionable_impacts, shap_importance_df, shap_actionable_values = counterfactuals(
            model=model,
            X_train=X_train,
            X_test=X_test,
            actionable_features=actionable_features,
            target_value=target_value,
            num_solutions=num_solutions,
            reduced_features_path=reduced_features_path,
            user_input=user_data_df.iloc[-1].to_dict()
            )

        if return_fig:
            plt.close() 
            return fig
        else:
            plt.show()

    if return_fig:
        return (fig)
    else:
        return (future_phq9, target_value, counterfactuals_list, user_data_df)
