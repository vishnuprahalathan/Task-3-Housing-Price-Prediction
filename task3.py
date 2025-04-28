
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_path = r"C:\Users\Vishnu Prahalathan\Desktop\Housing.csv"
data = pd.read_csv(file_path)

print("First 5 rows of the dataset:")
print(data.head())


print("\nChecking missing values:")
print(data.isnull().sum())

data = data.dropna()

print("\nAvailable columns in dataset:")
print(data.columns)


possible_targets = ['median_house_value', 'median_house_price', 'price', 'SalePrice']
target_column = None
for col in possible_targets:
    if col in data.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError("Target column like 'median_house_value' not found! Please check dataset columns.")

print(f"\nTarget column detected for prediction: {target_column}")


categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea', 'furnishingstatus']


binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})


data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)


cleaned_file_path = r"C:\Users\Vishnu Prahalathan\Desktop\Housing_Cleaned.csv"
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved successfully at: {cleaned_file_path}")


X = data.drop(target_column, axis=1)
y = data[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr_multiple = LinearRegression()
lr_multiple.fit(X_train_scaled, y_train)


y_pred_multiple = lr_multiple.predict(X_test_scaled)
print("\nMultiple Linear Regression Evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_multiple):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_multiple):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_multiple)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_multiple):.2f}")


coefficients = pd.DataFrame(lr_multiple.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coefficients)


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

importance = np.abs(lr_multiple.coef_)
feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance Ranking:")
print(feature_importance)


ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
print("\nRidge Regression R² Score:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
print("\nLasso Regression R² Score:", r2_score(y_test, lasso_pred))
