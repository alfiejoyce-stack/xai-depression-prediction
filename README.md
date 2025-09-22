# Explainable Machine Learning for Mental Health

This repository contains the code and results from my Final Year Individual Project (BEng Mechanical Engineering, Loughborough University).  
The project investigates methods for predicting depression severity using the NHANES dataset and focuses on the visual communication of complex machine learning outputs through Explainable AI (XAI).

## Project Overview
- Developed and tuned an XGBoost regression model to predict PHQ-9 depression severity scores.  
- Applied feature engineering, cross-validation, and oversampling to address class imbalance, overfitting, and high-dimensional data.  
- Incorporated SHAP (Shapley Additive Explanations) and counterfactual explanations to make model outputs interpretable and actionable.
- Designed two interactive Python dashboards:
  - Patient-facing: simple visualisations and practical guidance.  
  - Clinician-facing: detailed feature analysis using violin and parallel axis plots.  

## Repository Structure
```text
xai-depression-prediction/
├── README.md               <- Project overview
├── pip_install.txt        <- Dependencies for setup
│
├── data/                   <- Preprocessing and feature reduction
│   ├── calc_phq9.py
│   ├── feature_reduction.py
│   ├── feature_reduction_correlation.py
│   ├── Remove Unknown.py
│   └── README.md           <- Instructions for dataset handling
│
├── model/                  <- Final model and explainability
│   ├── main.py             <- Runs the full pipeline
│   ├── counterfactuals.py
│   ├── dashboard_visuals.py
│   ├── data_loader.py
│   ├── expert_dashboard.py
│   ├── feature_dependence_map.py
│   ├── heatmap.py
│   ├── main.py
│   ├── modelling.py
│   ├── non_expert_dashboard.py
│   ├── project_phq9_and_suggest.py
│   └── user_input_gui.py
│
└── results/                <- Outputs and visualisations
    ├── SHAP_feature_importance.png
    ├── Scatter_graph_phq9_vs_date.png
    ├── Feature_heatmap.png
    ├── dashboard_output.png
    └── Counterfactual_decision_tree.png
