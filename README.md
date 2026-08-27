# Task 1: Data Cleaning & Preprocessing (Titanic)

## Author
- **Name:** PURAB ROY  
- **Email:** roy.purab.28@gmail.com  

## Objective
Learn how to clean and prepare raw data for machine learning using Python, Pandas, NumPy, and Matplotlib/Seaborn.

## Steps Performed

1. Loaded the Titanic dataset and explored basic info (shape, columns, data types, missing values).
2. Handled missing values:
   - `Age`: median imputation  
   - `Embarked`: most frequent value  
   - `Cabin`: created binary feature `HasCabin`, then dropped `Cabin`
3. Selected features and target (`Survived`).
4. Encoded categorical features (`Sex`, `Embarked`) using one‑hot encoding.
5. Standardized numerical features using `StandardScaler`.
6. Detected outliers using boxplots on `Age`, `Fare`, and `SibSp`.
7. Removed outliers using the IQR rule.
8. Built a preprocessing pipeline ready for machine learning models.

## How to Run

1. Download the Titanic dataset (`titanic.csv`) if you want to run locally.
2. Open `task1_titanic_preprocessing.ipynb` in Google Colab or Jupyter.
3. Run all cells in order.

## Interview Questions (Short Answers)

1. **Types of missing data:** MCAR, MAR, MNAR.  
2. **Handling categorical variables:** One‑hot encoding (nominal), label/ordinal encoding (ordered).  
3. **Normalization vs standardization:**  
   - Normalization: rescales to [0, 1].  
   - Standardization: rescales to mean 0, variance 1.  
4. **Detecting outliers:** Boxplots, IQR rule, z‑scores.  
5. **Why preprocessing matters:** Models need clean, numeric, scaled inputs; preprocessing improves convergence and accuracy.  
6. **One‑hot vs label encoding:** One‑hot creates binary columns; label maps categories to integers.  
7. **Handling data imbalance:** Resampling (SMOTE, undersampling), class weights, threshold tuning, appropriate metrics.  
8. **Can preprocessing affect accuracy?** Yes—good preprocessing usually improves it; bad preprocessing can hurt it.
