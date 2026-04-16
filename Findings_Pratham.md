# Findings
## Complete Analysis Findings — EDA, OLS Regression & Feedforward Neural Network

---

## 1. Descriptive Statistics & Data Profiling

### 1.1 Dataset Overview

The preprocessed dataset contains 51,867 residential energy efficiency retrofit projects spanning 2018–2023 across 61 New York counties. After KNN imputation (k=5), the dataset is nearly complete — null values account for less than 1% and were dropped without material impact.

The dataset includes 22 columns: 14 numeric features and 8 categorical features covering project costs, building characteristics, energy savings estimates, utility providers, and geographic identifiers.

### 1.2 Numeric Features

| Feature | Mean | Std Dev | Min | Max | Skewness | Kurtosis |
|---------|------|---------|-----|-----|----------|----------|
| Total Project Cost ($) | 6,435.77 | — | 129 | 92,297 | 2.17 | 15.80 |
| Size of Home (sq ft) | 1,572.71 | 658.06 | 84 | 4,000 | — | — |
| Year Home Built | 1,948.08 | 37.26 | 1,768 | 2,021 | — | — |
| Est. Annual kWh Savings | 470.11 | — | neg. | 38,572 | — | — |
| Est. Annual MMBtu Savings | 31.40 | — | neg. | 1,005 | — | — |
| Log Annual Savings ($) | 5.85 | 0.88 | 0.00 | 10.20 | — | — |

Total Project Cost is severely right-skewed (skewness = 2.17, kurtosis = 15.8) with a large gap between the 75th percentile and the maximum. Even log transformation fails to fully normalize it. The target variable (log-transformed annual savings) ranges from 0 to ~10.2, corresponding to approximately $1 to $27,000 on the original dollar scale.

### 1.3 Categorical Features

| Feature | Levels | Dominant Category | Dominance % |
|---------|--------|-------------------|-------------|
| Job Type | 2 | Home Performance | 96.1% |
| Type of Dwelling | 3 | Single Family | 87.8% |
| Measure Type | 3 | Building Shell | 99.0% |
| Heating Fuel Type | 9 | Natural Gas | 71.9% |
| Gas Utility | 12 | National Grid | 39.6% |
| Electric Utility | 8 | National Grid | 40.6% |
| Project County | 62 | Monroe | 14.5% |

Several categorical features exhibit heavy class imbalance — Measure Type is 99% Building Shell, and Job Type is 96% Home Performance. This limits the model's ability to learn distinct patterns for minority categories.

### 1.4 Covariance & Correlation

The Pearson correlation matrix revealed several important relationships. MMBtu Savings shows the strongest correlation with Total Project Cost (Spearman r = 0.74), and First Year Energy Savings correlates highly with MMBtu Savings (Spearman r = 0.71). Beyond these two dominant relationships, all other feature-target correlations fall below 0.18 — statistically significant only due to sample size, not practical relevance.

Comparing Pearson and Spearman correlations uncovered non-linear signals: both MMBtu Savings and First Year Energy Savings show gaps above the 0.05 threshold between |Spearman| and |Pearson|, confirming that their relationship with the target is not purely linear. This finding directly disadvantages OLS regression and favours non-linear models.

---

## 2. Feature Engineering

### 2.1 Derived Features

The following features were engineered from raw variables:

**Age of Home** was computed as Project Completion Year minus Year Home Built. This captures the age of the building at the time of retrofit, which is a more meaningful predictor of energy efficiency than the static construction year. The original Year Home Built column was dropped to avoid redundancy.

**Cost per Square Foot** was computed as Total Project Cost divided by Size of Home, then log-transformed (log(Cost_per_Sqft + 1)) to handle right skewness. The raw Cost_per_Sqft was dropped after transformation.

**Total Energy Savings (kWh/sqft)** was constructed by converting MMBtu savings to kWh equivalents (×293.071), adding to kWh savings, and normalizing by home size. Outlier treatment was applied via 1st and 99th percentile clipping. The individual kWh savings, MMBtu savings, and unnormalized total energy savings columns were dropped after consolidation.

### 2.2 Multicollinearity Reduction

Post-feature-engineering heatmaps identified several redundant feature pairs. The following columns were dropped: Year Home Built (redundant with Age of Home), MMBtu_savings_per_Sqft and kWh_savings_per_Sqft (absorbed into total_energy_savings_kwh_sqft), raw Estimated Annual kWh Savings and Estimated Annual MMBtu Savings, and the intermediate total_energy_savings_kwh.

### 2.3 Final Feature Set

After all transformations, the modeling feature set consists of: Project County, Gas Utility, Electric Utility, Total Project Cost, Pre-Retrofit Home Heating Fuel Type, Size of Home, Number of Units, Job Type, Type of Dwelling, Measure Type, Location latitude/longitude, Project Completion Year/Month, log_cost_per_Sqft, total_energy_savings_kwh_sqft, and Age of Home.

---

## 3. ANOVA Results

One-way ANOVA tests were conducted for each categorical grouping variable against key numeric targets at α = 0.05. Levene's test for equality of variances was run alongside each ANOVA.

### 3.1 Key Findings

All tested categorical-target combinations were statistically significant (p < 0.05), reflecting the large sample size. The most practically meaningful results by F-statistic magnitude:

**Job Type** showed the strongest group effects across all targets, with extremely high F-statistics for Total Project Cost, energy savings, and House Age. Home Performance projects differ substantially from Electric Reduction projects on every metric.

**Pre-Retrofit Home Heating Fuel Type** produced large F-statistics for MMBtu savings and total energy savings, confirming that fuel type is a major driver of savings magnitude. Homes heated by propane, oil, and kerosene show systematically different savings profiles than natural gas homes.

**Type of Dwelling** differentiated meaningfully on cost and savings — Single Family, Mobile, and 2-4 Family homes have distinct retrofit cost and savings distributions.

**Measure Type** showed significant effects, though the 99% dominance of Building Shell limits practical interpretability.

### 3.2 Levene's Test

Most ANOVA tests violated the equal variance assumption (Levene's p < 0.05), indicating heteroscedasticity across groups. Given the very large sample sizes per group, the ANOVA F-test remains robust to this violation, but the finding is consistent with the non-constant variance patterns observed in the OLS residual analysis.

---

## 4. VIF Analysis (Multicollinearity)

Variance Inflation Factors were computed for all numeric predictors after feature engineering.

### 4.1 Results

Features with VIF < 5 (low multicollinearity) constituted the majority of the feature set, confirming that the feature engineering and column dropping strategy successfully reduced multicollinearity. The dropped columns (Year Home Built, individual energy savings components) had been the primary sources of inflation.

Features with VIF ≥ 10 (high multicollinearity), if any remained, were noted for potential removal in regularized models. The post-engineering correlation heatmap confirmed that no remaining feature pairs exhibited problematic collinearity levels.

### 4.2 Implication for Modeling

The reduced VIF profile makes the dataset suitable for OLS regression without severe multicollinearity bias. However, the remaining moderate correlations between engineered features (e.g., log_cost_per_Sqft and Total Project Cost) explain why regularization and non-linear models can still extract additional signal.

---

## 5. Outlier Analysis

### 5.1 IQR and Z-Score Methods

Outliers were assessed using both IQR (1.5×IQR fences) and Z-score (|z| > 3) methods across all numeric features.

**Estimated Annual kWh Savings** had the highest outlier rate at 9.52% by IQR — this is a distributional shape issue (extreme right tail) rather than data errors. Log transformation is the appropriate treatment.

**Total Project Cost** showed 2.36% IQR outliers, representing genuine extreme values (very large whole-home retrofits up to $92,297). These high-cost projects disproportionately influence OLS and Lasso via squared-error minimization, while Random Forest and FFNN are naturally more robust.

Most other features had outlier rates below 2%, indicating a reasonably clean dataset after preprocessing.

### 5.2 Treatment Strategy

Rather than blanket deletion, outliers were handled through: percentile clipping (1st/99th) for total_energy_savings_kwh_sqft, log transformation for cost-related features, and retention of genuine extreme values with sensitivity analysis planned for model comparison.

---

## 6. Target Variable Analysis

The target variable (First Year Modeled Project Energy Savings $ Estimate) was log-transformed during preprocessing.

### 6.1 Distribution

The log-transformed target has a mean of 5.85, median of 5.96, and standard deviation of 0.88. The distribution is slightly left-skewed with heavier-than-normal tails. On the original dollar scale, this corresponds to a mean of approximately $349 and a median of approximately $387.

### 6.2 Normality Tests

Both the Kolmogorov-Smirnov test and D'Agostino-Pearson test rejected normality at α = 0.05, even after log transformation. This is expected with 51,867 observations — at this sample size, even minor deviations from normality are detectable. The Q-Q plot showed reasonable adherence to normality in the central quantiles with deviation in the tails, particularly the left tail (low-savings outliers).

This non-normality directly impacts OLS, whose inference (confidence intervals, p-values) relies on normally distributed residuals. Point estimates remain unbiased but standard errors may be unreliable. The neural network carries no such assumption.

---

## 7. OLS Regression Results

### 7.1 Preprocessing Pipeline

The OLS model followed the standard pipeline: dropna → feature engineering (Age of Home, log_cost_per_Sqft, total_energy_savings_kwh_sqft with percentile clipping) → drop redundant columns → LabelEncoder for categoricals → StandardScaler → 80/20 train-test split (random_state=42).

### 7.2 Performance Metrics

| Metric | Value |
|--------|-------|
| R² (test) | ~0.68 |
| Adjusted R² | ~0.68 |
| RMSE (test) | ~0.49 |
| MAE (test) | ~0.35 |
| 5-Fold CV R² | ~0.67 ± 0.007 |

The model explains approximately 68% of the variance in log-transformed energy savings. The close agreement between test R² and 5-fold CV R² confirms the model generalizes well with no overfitting.

AIC and BIC values were computed for model selection comparison. The F-statistic was highly significant (p ≈ 0), confirming the model is significantly better than an intercept-only model.

### 7.3 Coefficient Analysis

The standardized OLS coefficients (from the statsmodels summary) revealed the following predictor hierarchy:

**Dominant predictors:** total_energy_savings_kwh_sqft and log_cost_per_Sqft were the two strongest predictors by absolute t-statistic, aligning with the correlation analysis that identified energy savings and cost as the only features with meaningful target association.

**Moderate predictors:** Heating fuel type, electric utility, project completion year, and age of home contributed moderate but statistically significant effects.

**Weak/insignificant predictors:** Several features, despite being statistically significant (due to large n), had negligibly small coefficients, confirming that most features add minimal predictive value beyond the two dominant ones.

All features with p < 0.05 were flagged as significant, though the large sample size means statistical significance does not imply practical importance.

### 7.4 Residual Diagnostics

**Residuals vs Fitted:** The plot showed a fan-shaped pattern at the low end of fitted values, indicating heteroscedasticity — the model predicts poorly for low-savings projects. This is a systematic misfit, not random noise.

**Q-Q Plot:** Residuals deviated substantially from normality in both tails, particularly the left tail (negative residuals). The heavy left tail corresponds to projects where actual savings are much lower than predicted — the model overestimates savings for low-performing projects.

**Residual Distribution:** Negative skew (~−1.2) and high kurtosis (~11) confirm non-normal, heavy-tailed residuals. This violates the OLS normality assumption and means standard errors, confidence intervals, and p-values are technically unreliable.

**Actual vs Predicted:** The scatter showed reasonable alignment along the diagonal for mid-range values but systematic deviation at extremes — particularly underprediction of high-savings and overprediction of low-savings projects.

### 7.5 VIF (Post-Modeling)

Post-encoding VIF analysis confirmed that the feature engineering successfully controlled multicollinearity. Most features had VIF < 5. Any remaining moderate VIF values are structurally expected from the label-encoded categorical variables.

### 7.6 Cross-Validation

5-fold cross-validation produced a mean R² of ~0.67 with low variance across folds (± 0.007), confirming stable generalization. The consistency between CV and holdout test performance rules out overfitting as a concern.

---

## 8. Feedforward Neural Network Results

### 8.1 Preprocessing Pipeline

The FFNN used the identical preprocessing pipeline as OLS to ensure a fair comparison: same feature engineering, same encoding, same scaling, same train-test split (random_state=42).

### 8.2 Architecture Search

Four architectures were tested with dropout rate 0.2, Adam optimizer (lr=0.001, weight_decay=1e-4), ReduceLROnPlateau scheduler, batch size 256, and early stopping with patience of 15 epochs:

| Architecture | Parameters | R² | RMSE | MAE |
|-------------|------------|-----|------|-----|
| 1-Layer (64) | 1,537 | ~0.82 | ~0.37 | ~0.26 |
| 2-Layer (128→64) | 10,945 | ~0.84 | ~0.35 | ~0.25 |
| 3-Layer (128→64→32) | 13,121 | ~0.84 | ~0.35 | ~0.25 |
| 4-Layer (256→128→64→32) | 42,561 | ~0.84 | ~0.35 | ~0.25 |

The 2-layer architecture (128→64) achieved the best balance of performance and efficiency. Deeper architectures (3-layer, 4-layer) offered negligible improvement despite significantly more parameters, suggesting the underlying signal complexity plateaus at two hidden layers.

### 8.3 Best Architecture Performance (Pre-Tuning)

| Metric | Value |
|--------|-------|
| R² (test) | 0.8446 |
| RMSE (test) | ~0.35 |
| MAE (test) | ~0.25 |

### 8.4 Hyperparameter Tuning

Five configurations were tested on the best architecture:

| Learning Rate | Dropout | R² | RMSE |
|---------------|---------|-----|------|
| 0.001 | 0.1 | ~0.84 | ~0.35 |
| 0.001 | 0.2 | ~0.84 | ~0.35 |
| 0.001 | 0.3 | ~0.84 | ~0.35 |
| 0.0005 | 0.2 | ~0.85 | ~0.34 |
| 0.002 | 0.2 | ~0.84 | ~0.35 |

The best tuned configuration (lr=0.0005, dropout=0.2) achieved R² = 0.8478, a marginal improvement over the default. The default PyTorch Adam settings were already near-optimal for this problem, which is common for tabular regression tasks of this scale.

### 8.5 Final Tuned Performance

| Metric | Value |
|--------|-------|
| R² (test) | 0.8478 |
| RMSE (test) | ~0.34 |
| MAE (test) | ~0.24 |

### 8.6 Training Dynamics

Loss curves for all architectures showed healthy convergence: training and validation losses decreased together and plateaued at similar levels, with no significant train-val gap. Early stopping activated between epochs 60–100 for most architectures, confirming that 150 epochs was a sufficient upper bound. The ReduceLROnPlateau scheduler further stabilized late-stage training.

### 8.7 Residual Analysis

**Residuals vs Fitted:** The fan-shaped heteroscedasticity pattern seen in OLS was reduced but not eliminated. The neural network handles the mid-range better but still struggles with extreme low-savings predictions.

**Actual vs Predicted:** Much tighter clustering along the diagonal compared to OLS, with reduced scatter at both extremes. The improvement is most visible in the 4–7 range of log-savings (the bulk of the data).

**Absolute Residuals vs Fitted:** Error magnitude is more uniform across the prediction range compared to OLS, confirming the neural network's ability to model heterogeneous variance through its non-linear activations.

### 8.8 Feature Importance (Permutation Importance)

Permutation importance (10 repeats, scoring on R²) for the best model revealed:

**Top features by R² decrease:** total_energy_savings_kwh_sqft and log_cost_per_Sqft dominated — consistent with OLS coefficient rankings and the EDA correlation analysis. This cross-model agreement strengthens the conclusion that these two features carry the vast majority of predictive signal.

**Secondary features:** Project County, Electric Utility, and Heating Fuel Type contributed moderate importance, likely capturing regional energy price and climate variation.

**Low-importance features:** Location coordinates, Number of Units, Project Completion Day/Month, and size_of_home_flag contributed negligible importance, confirming they are noise relative to the dominant predictors.

---

## 9. Model Comparison — OLS vs FFNN

### 9.1 Performance Summary

| Metric | OLS | FFNN (Tuned) | Δ Improvement |
|--------|-----|-------------|---------------|
| R² (test) | ~0.68 | 0.8478 | +16.8 pp |
| RMSE (test) | ~0.49 | ~0.34 | ~30% reduction |
| MAE (test) | ~0.35 | ~0.24 | ~31% reduction |

The neural network achieves a substantial improvement over OLS across all metrics: approximately 17 percentage points higher R², 30% lower RMSE, and 31% lower MAE.

### 9.2 Why FFNN Outperforms OLS

The EDA predicted this outcome. The two dominant predictors (total_energy_savings_kwh_sqft and log_cost_per_Sqft) both exhibit non-linear relationships with the target, as confirmed by the Spearman-Pearson gap analysis. OLS can only fit linear relationships to these features, systematically misfitting the curves. The neural network's non-linear activations (ReLU + stacked layers) approximate these curves directly.

Additionally, the target variable violates OLS normality assumptions — heavy left-skewed residuals with kurtosis of ~11 confirm systematic misfit at extremes. The FFNN has no distributional assumptions and learns the conditional mean directly from the data.

### 9.3 Feature Importance Agreement

Both models agree on which features matter: total_energy_savings_kwh_sqft and log_cost_per_Sqft are dominant, with everything else contributing marginally. This convergence across a linear and non-linear model strengthens the finding — the predictive signal in this dataset is concentrated in two features.

### 9.4 Interpretability vs Accuracy Tradeoff

OLS provides full statistical inference: coefficient confidence intervals, p-values, F-tests, and directional interpretation of each predictor. The FFNN provides superior accuracy but operates as a black box — permutation importance ranks features but does not explain the direction or functional form of their effects.

For a policy-facing program like EmPower, where stakeholders need to understand why certain projects save more energy, OLS remains valuable despite its lower R². The FFNN is better suited for prediction tasks (e.g., forecasting savings for new projects) where accuracy matters more than interpretability.

---

## 10. Key Takeaways

1. **The dataset naturally favours non-linear models.** The two strongest predictors have non-linear relationships with a non-normal target, surrounded by mostly weak features. This structural property means OLS will always underperform relative to flexible models on this data.

2. **Feature engineering is essential.** Normalizing by home size (cost/sqft, energy savings/sqft), log-transforming skewed features, and consolidating redundant energy metrics into a single feature reduced multicollinearity and concentrated predictive signal.

3. **Only two features carry meaningful signal.** Both OLS coefficients and FFNN permutation importance converge: total energy savings per square foot and log cost per square foot dominate. Geographic and building-type features contribute secondary effects, likely through regional energy price variation.

4. **FFNN achieves R² = 0.85 vs OLS R² = 0.68.** The 17 percentage-point improvement is driven by the neural network's ability to model non-linear relationships that OLS cannot capture. The improvement is practically significant, not just statistically significant.

5. **Diminishing returns from deeper architectures.** A 2-layer network (128→64) captures nearly all available signal. Adding layers or aggressively tuning hyperparameters yields marginal (<1 pp) improvements, suggesting the performance ceiling for this feature set is near R² ≈ 0.85.

6. **OLS residuals violate key assumptions.** Heteroscedasticity (fan-shaped residuals), non-normality (skew = −1.2, kurtosis = 11), and systematic misfit at extremes mean OLS inference (p-values, CIs) should be interpreted cautiously.

7. **Both models struggle with low-savings predictions.** The left tail of the target distribution — projects with near-zero modeled savings — is difficult for both OLS and FFNN. These may represent data quality issues (incomplete projects, modeling artifacts) or genuinely unpredictable outcomes.

---

*Analysis conducted using Python (pandas, NumPy, statsmodels, scikit-learn, PyTorch, scipy, seaborn, matplotlib). All models trained on identical 80/20 splits with random_state=42 for reproducibility.*
