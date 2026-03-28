# Predicting Energy Savings in Low-Income Residential Retrofit Projects  
### A comparative analysis of OLS, ensemble techniques, and Neural Networks on regression tasks  

---

## Preprocessing Step  
---

**Number of records:** 51,934  
**Number of features:** 20  

---

## Target Variable Transformation  
---

The target variable was **left-skewed**, violating the normality assumption required for linear regression.  

To address this, a **log(x + 1) transformation** was applied to:
- Handle zero values  
- Normalize the distribution  
- Improve model performance  

---

## Handling Missing Values  
---

### Gas Utility  

Missing values were imputed using a **hierarchical mode-based approach**:  
**Project County → Electric Utility → Mode(Gas Utility)**  

---

### Year Built (Updated Approach)  

Missing values in **Year Home Built** were imputed using a **K-Nearest Neighbors (KNN) based approach**.

#### Methodology  

- Removed non-informative and leakage-prone features (including target variable)  
- Encoded categorical variables and standardized numerical features  
- Applied KNN imputation with:
  - **k = 3 neighbors**
  - **k = 5 neighbors**  

#### Model Selection  

- Compared both imputed datasets  
- Selected the dataset with **lower standard deviation**, ensuring minimal distribution distortion  

---

## Insights from KNN Imputation  
---

- **Improved Data Realism**  
  KNN preserves local patterns, resulting in more realistic year assignments compared to median-based filling  

- **Reduced Distribution Distortion**  
  Avoids artificial clustering around central values and maintains a smoother distribution  

- **Better Feature Relationships**  
  Year Built now aligns better with:
  - Project Cost  
  - Size of Home  
  - Job Type  

- **Handling of Edge Cases**  
  Extreme values (older homes) are better preserved instead of being over-smoothed  

- **Trade-off**  
  Increased computational cost, but significantly improved imputation quality  

---

## Data Standardization for House Size  
---

A flag-based approach was applied:

| Flag | Description |
|------|------------|
| 0 | Exact value present |
| 1 | < 800 sq ft |
| 2 | > 800 sq ft |
| 3 | < 4000 sq ft |
| 4 | > 4000 sq ft |

This preserves boundary-based information while enabling numeric modeling.

---

## Data Type Conversion  
---

The following columns were converted:

- `Reporting Period` → datetime  
- `Project Completion Date` → datetime  
- `Size Of Home` → numeric  
- `Estimated Annual MMBtu Savings` → numeric  
- `Estimated Annual kWh Savings` → numeric  

---

## Feature Engineering (Geospatial Data)  
---

Extracted features:

- City  
- State  
- Zipcode  
- Latitude  
- Longitude  

These features help capture regional variations in energy consumption and savings.

---

## Removing Redundant Features  
---

The following columns were removed as duplicates:

- `Location_1_city`  
- `Location_1_state`  
- `Location_1_zipcode`  

---

## Final Dataset  
---

The preprocessed dataset was saved as:

```python
df = pd.read_csv("Preprocessed_dataset_knn(5neighbour).csv")