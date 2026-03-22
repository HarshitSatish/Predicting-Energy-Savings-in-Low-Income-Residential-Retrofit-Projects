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

**Before Transformation:**  
![Before](https://github.com/user-attachments/assets/07800376-c4eb-402c-9106-2ba5d258a46d)

**After Log Transformation:**  
![After](https://github.com/user-attachments/assets/ba2c72e3-f5f9-49c4-ac0d-52d8812cbe53)

---

## Handling Missing Values  
---

### Gas Utility  

Missing values were imputed using a **hierarchical mode-based approach**:  
**Project County → Electric Utility → Mode(Gas Utility)**  

---

### Year Built  

A flag variable was created to preserve data meaning:

| Flag | Description |
|------|------------|
| 0 | Year present |
| 1 | Homes built before 1800 |
| 2 | Missing values |

- Missing values (flag = 2) were filled using the **median year**  
- `<1800` values were preserved using the flag  

---

## Data Standardization for House Size  
---

A similar flag-based approach was applied:

| Flag | Description |
|------|------------|
| 0 | Exact value present |
| 1 | < 800 sq ft |
| 2 | > 800 sq ft |
| 3 | < 4000 sq ft |
| 4 | > 4000 sq ft |

This preserves **boundary-based information** while enabling numeric modeling.

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

These features help capture **regional variations in energy consumption and savings**.

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
df = pd.read_csv("Preprocessed_dataset.csv")
