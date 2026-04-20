PREDICTING ENERGY SAVINGS IN LOW-INCOME RESIDENTIAL RETROFIT PROJECTS A
COMPARATIVE ANALYSIS OF OLS, ENSEMBLE TECHNIQUES, AND NEURAL NETWORKS

------------------------------------------------------------------------

PROJECT OVERVIEW

This project predicts annual energy savings (in $) for low-income
residential retrofit programs using machine learning models. Accurate
prediction helps optimize funding allocation and improve energy
efficiency strategies.

------------------------------------------------------------------------

OBJECTIVE

-   Predict First Year Modeled Project Energy Savings ($)
-   Compare OLS, Ridge, Lasso, Random Forest, XGBoost, Neural Networks
-   Perform preprocessing, feature engineering, encoding, and scaling
-   Optimize models using hyperparameter tuning
-   Evaluate using RMSE and R²

------------------------------------------------------------------------

DATASET

Source: NYSERDA Retrofit Program Records: 51,934 Features: 20

------------------------------------------------------------------------

PREPROCESSING

TARGET TRANSFORMATION Applied log(x + 1) transformation to handle
skewness and stabilize variance.

MISSING VALUES Gas Utility: Hierarchical imputation (County → Electric
Utility → Mode)

Year Home Built: KNN Imputation (k=3, k=5) Selected dataset with lower
variance to preserve distribution. Target leakage avoided.

------------------------------------------------------------------------

FEATURE ENGINEERING

-   Age of Home = Completion Year − Year Built
-   Cost per Sqft (log transformed)
-   Energy Savings per Sqft
-   Temporal features (Year, Month, Day)
-   Geospatial features (City, State, Zip, Latitude, Longitude)

------------------------------------------------------------------------

ENCODING

-   Frequency Encoding: County, City
-   One-Hot Encoding: Job Type, Dwelling, Measure Type, Fuel Type

------------------------------------------------------------------------

SCALING

-   StandardScaler applied

------------------------------------------------------------------------

MODELS

-   OLS, Linear Regression
-   Ridge, Lasso
-   Decision Tree, Random Forest, XGBoost
-   Neural Networks

------------------------------------------------------------------------

HYPERPARAMETER TUNING

GridSearchCV improved R² from 84.8% to 85.8%

------------------------------------------------------------------------

EVALUATION

Metrics: - RMSE - R²

Visualizations: - Actual vs Predicted - RMSE comparison - Residual plots

------------------------------------------------------------------------

INSIGHTS

-   Feature engineering improved performance
-   Tree models captured non-linearity
-   Cost per sqft and home size are key drivers

------------------------------------------------------------------------

CONCLUSION

This project demonstrates the effectiveness of combining statistical and
machine learning models for predicting energy savings.

------------------------------------------------------------------------

AUTHOR

Harshit Satishkumar, MS Data Science, Northeastern University
Harshith Vasantha Rajkumar, MS Data Science, Northeastern University
Pratham Paras Patel, MS Data Science, Northeastern University

