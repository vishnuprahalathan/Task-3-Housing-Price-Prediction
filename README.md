# Task-3-Housing-Price-Prediction

This project aims to predict housing prices based on various independent features using multiple regression techniques.  
It covers complete data cleaning, encoding, model training, evaluation, and feature importance analysis.

---

Project Overview

- Dataset: `Housing.csv` containing features like area, bedrooms, bathrooms, air conditioning, etc.
- Objective: Predict the `price` of houses based on provided features using machine learning regression techniques.

---

Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (sklearn)

---
Project Workflow

1. Data Loading
- Loaded dataset from `C:\Users\Vishnu Prahalathan\Desktop\Housing.csv`.

2. **Data Cleaning & Preprocessing
- Dropped missing value rows.
- Identified and selected `price` as the target variable.
- Encoded categorical columns:
  - Converted binary columns (`yes`/`no`) into `1` and `0`.
  - One-hot encoded multi-category column `furnishingstatus`.

3. Saving Cleaned Dataset
- Saved the cleaned dataset as `Housing_Cleaned.csv` on Desktop.

4. Feature Scaling
- Applied `StandardScaler` to normalize the features before feeding into models.

5. Model Training
- Built a Multiple Linear Regression model.
- Additionally built Ridge and Lasso regression models for comparison.

6. Model Evaluation
Evaluated models using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

7. Feature Importance Analysis
- Identified the most influential features on housing price prediction.

 8. Visualizations
- Correlation heatmap between all features.

---

Important Notes

- Feature scaling was critical because Ridge and Lasso are sensitive to magnitudes.
- Encoding categorical variables was necessary to avoid `ValueError` during scaling.
- One-hot encoding avoids introducing false order among categories like `furnishingstatus`.
- Ridge and Lasso helped explore regularization impact on model performance.

---



