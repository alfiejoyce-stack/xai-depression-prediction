from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def tune_hyperparameters(X_train, y_train, param_grid, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV.

    - X_train: Training feature data.
    - y_train: Training target data.
    - param_grid: Hyperparameter grid for searching.
    - cv: Number of cross-validation folds.

    Returns:
    - best_model: The best model with optimized hyperparameters.
    """
    model = XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best Hyperparameters: {best_params}")
    return best_model, best_params

def preprocess_data(X, y, threshold_corr=0.5, threshold_var=0.01, 
                   threshold_missing=0.5, n_top_features=20,
                   save_path='reduced_training_data.csv'):
    """
    Enhanced preprocessing pipeline:
    1. Handles missing values
    2. Removes low-variance and correlated features (preserving important ones)
    3. Selects top features using feature importance
    4. Saves processed data
    5. Returns processed X and y
    """

    assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
    assert len(X) == len(y), "X and y must have same length"
    
    print(f"\n{' Preprocessing Started ':=^80}")
    print(f"Original shape: {X.shape}")
    
    y = pd.Series(y, name='PHQ9_Target') if isinstance(y, np.ndarray) else y.rename('PHQ9_Target')

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_lambda': [5, 10],
    }

    # Get the tuned model
    model, best_params = tune_hyperparameters(X, y, param_grid)

    # Drop specific columns manually
    if 'WTMEC2YR' in X.columns:
        X = X.drop(columns=['WTMEC2YR'])
        print("Manually dropped 'WTMEC2YR' from the dataset")
    if 'RIDSTATR' in X.columns:
        X = X.drop(columns=['RIDSTATR'])
        print("Manually dropped 'RIDSTATR' from the dataset")

    # Drop columns that start with 'FNQ' or 'BPAOARM'
    cols_to_drop = [col for col in X.columns if col.startswith('FNQ') or col.startswith('BPAOARM')or col.startswith('LUAPNME')]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        print(f"Manually dropped columns starting with 'FNQ' or 'BPAOARM' or 'LUAPNME': {', '.join(cols_to_drop)}")

    # Drop columns with too many missing values
    missing_percentage = X.isnull().mean()
    cols_to_drop_missing = missing_percentage[missing_percentage > threshold_missing].index
    X = X.drop(columns=cols_to_drop_missing)
    print(f"1. Dropped {len(cols_to_drop_missing)} columns with >{threshold_missing*100}% missing values")

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print("2. Imputed missing values using median")

    # Drop low-variance features
    low_var_cols = X_imputed.var()[X_imputed.var() < threshold_var].index
    X_imputed = X_imputed.drop(columns=low_var_cols)
    print(f"3. Dropped {len(low_var_cols)} low-variance features")

    # Drop highly correlated features (keep most important)
    print("4. Checking for correlated features...")
    # Model to estimate importance
    temp_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    temp_rf.fit(X_imputed, y)
    temp_importance = pd.Series(temp_rf.feature_importances_, index=X_imputed.columns)

    # Correlation matrix
    corr_matrix = X_imputed.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            if upper_triangle.loc[row, col] > threshold_corr:
                drop_col = row if temp_importance[row] < temp_importance[col] else col
                to_drop.add(drop_col)

    X_imputed = X_imputed.drop(columns=list(to_drop))
    print(f"4. Dropped {len(to_drop)} correlated features using importance-aware filtering")

    # Select top N features
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_imputed, y)
    importance_df = pd.DataFrame({'feature': X_imputed.columns, 'importance': rf.feature_importances_})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    top_features = importance_df.head(n_top_features)['feature'].tolist()
    X_reduced = X_imputed[top_features]

    # Save reduced data
    reduced_data = pd.concat([X_reduced, y], axis=1)
    reduced_data.to_csv(save_path, index=False)

    print(f"Final shape: {X_reduced.shape}")
    print(f"Saved to: {save_path}")

    if save_path:
        with open(save_path.replace('.csv', '_features.txt'), 'w') as f:
            for feat in top_features:
                f.write(f"{feat}\n")


    return X_reduced, y



def train_model(X, y):
    """
    Complete training pipeline with:
    - Data preprocessing
    - Stratified splitting
    - Model training with cross-validation
    - Performance evaluation
    """
    # Preprocess data
    X_processed, y_processed = preprocess_data(X, y)
    
    # Stratified splitting 
    y_binned = pd.cut(y_processed, 
                     bins=[-1, 4, 9, 14, 28], 
                     labels=["0-4", "5-9", "10-14", "15+"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, 
        y_processed, 
        test_size=0.2, 
        stratify=y_binned, 
        random_state=42
    )
    
    # Verify stratification
    print("\n{' Data Distribution ':=^80}")
    dist_df = pd.DataFrame({
        'Train': pd.cut(y_train, bins=[-1, 4, 9, 14, 28]).value_counts(normalize=True),
        'Test': pd.cut(y_test, bins=[-1, 4, 9, 14, 28]).value_counts(normalize=True)
    })
    print(dist_df)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        reg_lambda=5,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1
    )


    # Cross-validation 
    print("\n{' Cross-Validation ':=^80}")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=cv, scoring='r2', n_jobs=-1)
    print(f"Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Training
    print("\n{' Model Training ':=^80}")

    model.fit(
        X_train, y_train,
        verbose=10, 
    )

    # Evaluation
    print("\n{' Final Evaluation ':=^80}")
    test_pred = model.predict(X_test)
    print(f"Test R²: {r2_score(y_test, test_pred):.3f}")
    print(f"Test MSE: {mean_squared_error(y_test, test_pred):.3f}")
    
    # Feature importance
    print("\n{' Feature Importance ':=^80}")
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10).to_string(index=False))

    return model, X_test, y_test, X_train, y_train
