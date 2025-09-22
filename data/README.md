# Data Preprocessing and Feature Engineering

This folder contains scripts used to prepare and process the NHANES dataset for model training and analysis.  
The raw NHANES data files are **not included** in this repository due to size and licensing restrictions.  
They can be downloaded from the [CDC NHANES website](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023/).

## Workflow

The scripts in this folder should be run in sequence:

1. **Remove Unknown.py**  
   - Replaces string values marked as "Unknown" with `NaN`.  
   - Imputes missing numeric values with the median of the column.  
   - Imputes missing non-numeric values with the placeholder "Unknown".  
   - Produces a cleaned dataset with consistent handling of missing values.

2. **calc_phq9.py**  
   - Calculates PHQ-9 depression severity scores by summing DPQ-related questionnaire responses.  
   - Adds a new column `PHQ9_Score`.

3. **feature_reduction.py**  
   - Uses a Random Forest regressor to rank features by importance.  
   - Selects the top 50 most important features.  
   - Saves a reduced dataset containing only these predictors.

4. **feature_reduction_correlation.py**  
   - Further reduces the feature set by removing highly correlated variables (threshold = 0.7).  
   - Produces the final dataset ready for model training.

## Expected Outputs

- `merged_all_seqn_no_unknowns.csv` → cleaned dataset.  
- `training_data_phq9.csv` → dataset with PHQ-9 scores.  
- `reduced_features.csv` → dataset with top features.  
- `reduced_features_after_correlation_reduction.csv` → final feature-reduced dataset.

## Notes

- Update file paths in scripts if your folder structure differs.  
- Place the NHANES dataset in this folder (or a consistent location) before running the scripts.
